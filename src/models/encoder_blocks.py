import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, SeparableConv1D
from src import layers


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, dropout, sublayer, total_sublayers, **kwargs):
        """ Convolutional Block

            Follows paper "QANet" (https://arxiv.org/pdf/1804.09541.pdf) and constructs a block of
            layer_norm -> convolutional -> residual.

            Args:
                filters: Number of filters in each block.
                kernel_size: Width of the kernel in this block.
                dropout: Fraction of input units to drop
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
                heads: Number of attention heads to use.

        """
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()

        self.seperable_conv = SeparableConv1D(filters=filters,
                                              kernel_size=kernel_size,
                                              padding='same',
                                              activation='relu')

        self.layer_dropout = layers.SublayerConnection(dropout, sublayer, total_sublayers)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        residual = x
        x = self.layer_norm(x)
        x = self.seperable_conv(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class SelfAttentionBlock(tf.keras.Model):
    def __init__(self, filters, sublayer, total_sublayers, heads=8, dropout=0.0, **kwargs):
        """ Self-Attention Block

            Follows paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf, section 3.3) with
            Layer Dropout following "QANet" (https://arxiv.org/pdf/1804.09541.pdf).

            Args:
                filters: Number of filters in each block.
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
                heads: Number of attention heads to use.
                dropout: Fraction of input units to drop
        """
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()
        self.multi_head_attention = layers.MultiHeadAttention(filters,
                                                              num_heads=heads,
                                                              dropout=dropout,
                                                              name='multi_head_attention')
        self.layer_dropout = layers.SublayerConnection(dropout, sublayer, total_sublayers)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        residual = x
        x = self.layer_norm(x)
        x = self.multi_head_attention([x, x], training=training, mask=mask)
        x = self.layer_dropout([x, residual], training=training)
        return x


class FeedForwardBlock(tf.keras.Model):
    def __init__(self, filters, dropout, sublayer, total_sublayers, ff_mul=1.0, **kwargs):
        """ Feed forward block

            Follows paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf, section 3.3) with
            Layer Dropout following "QANet" (https://arxiv.org/pdf/1804.09541.pdf).

            Args:
                filters: Number of filters in each block.
                dropout: Fraction of input units to drop.
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
                ff_mul: Feed-Forward Multiplier, increases the number of units in the first feed-forward
                                layer by a float multiplier.
        """
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()
        self.feed_forward = layers.FeedForwardLayer(filters, ff_mul=ff_mul, dropout=dropout)
        self.layer_dropout = layers.SublayerConnection(dropout, sublayer, total_sublayers)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        residual = x
        x = self.layer_norm(x)
        x = self.feed_forward(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class EncoderBlock(tf.keras.Model):
    def __init__(self, conv_layers, kernel_size, block_number=0, total_blocks=1,
                 filters=128, heads=8, dropout=0.1, ff_mul=1.0, **kwargs):
        """ Builds an encoder block.

            Encoder block from the paper "QANet" (https://arxiv.org/pdf/1804.09541.pdf, section 2.2), it is roughly
            equivalent to a transformer block from "Attention is all you Need" (https://arxiv.org/pdf/1706.03762.pdf,
            section 3), the main differences are the separable convolutions before the self-attention layer and
            Layer Dropout.

            Args:
                conv_layers: Number of convolutional layers in this block.
                kernel_size: Width of the kernel in this block.
                block_number: This blocks position within the stack.
                total_blocks: Total blocks within the stack.
                filters: Number of filters in each block.
                heads: Number of attention heads to use.
                dropout: Fraction of input units to drop in all dropout layers within this block.
                ff_mul: Feed-Forward Multiplier, increases the number of units in the first feed-forward
                        layer of the FeedForwardBlock by a float multiplier.
        """
        super(EncoderBlock, self).__init__(**kwargs)
        # These Ids and counts are for determining layer dropout, higher layers == more chance of dropout
        self.block_start_id = (block_number * (conv_layers + 2)) + 1  # Start from one
        self.self_attention_id = (self.block_start_id + conv_layers)
        self.feed_forward_id = self.self_attention_id + 1
        self.total_sub_layers = (conv_layers + 2) * total_blocks

        # Each block has 4 components, Position encoding, n Conv Blocks, a self-attention block and a feed
        # forward block with residual connections between them (Implemented as part of the block)
        self.position_encoding = layers.PositionEncoding()
        # Can have n convs, create a list of conv blocks to iterate through incrementing their block_ids.
        self.conv_layers = [ConvBlock(filters=filters,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      sublayer=(self.block_start_id + i),
                                      total_sublayers=self.total_sub_layers,
                                      name='conv_block_%d' % (self.block_start_id + i)) for i in range(conv_layers)]

        self.self_attention = SelfAttentionBlock(filters,
                                                 heads=heads,
                                                 sublayer=self.self_attention_id,
                                                 total_sublayers=self.total_sub_layers,
                                                 dropout=dropout,
                                                 name='self_attention_%d' % self.self_attention_id)

        self.feed_forward = FeedForwardBlock(filters, dropout,
                                             ff_mul=ff_mul,
                                             sublayer=self.feed_forward_id,
                                             total_sublayers=self.total_sub_layers,
                                             name='feed_forward_%d' % self.feed_forward_id)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        x = self.position_encoding(x, training=training, mask=mask)
        for conv in self.conv_layers:
            x = conv(x, training=training, mask=mask)
        x = self.self_attention(x, training=training, mask=mask)
        x = self.feed_forward(x, training=training, mask=mask)
        return x


class EncoderBlockStack(tf.keras.Model):
    def __init__(self, blocks, conv_layers, kernel_size, filters=128, heads=8, dropout=0.1, ff_mul=1.0, **kwargs):
        """ Builds a stack of encoder blocks and handles input projection + output dropout.

            Wrapper around EncoderBlock that includes functionality for optional input projection,
            output dropout and creating multiple blocks with the same parameters.

            Args:
                blocks: Number of blocks in this stack.
                conv_layers: Number of convolutional layers in each block.
                kernel_size: Width of the kernels in each block.
                filters: Number of filters in each block.
                heads: Number of attention heads to use.
                dropout: Fraction of input units to drop in all dropout layers within this stack.
                ff_mul: Feed-Forward Multiplier, increases the number of units in the first feed-forward
                        layer of the FeedForwardBlock by a float multiplier.
        """
        super(EncoderBlockStack, self).__init__(**kwargs)
        self.hidden_size = filters

        self.projection = Conv1D(filters,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 activation=None)

        self.blocks = [EncoderBlock(conv_layers=conv_layers, kernel_size=kernel_size,
                                    filters=filters, heads=heads,
                                    dropout=dropout, block_number=i, total_blocks=blocks,
                                    ff_mul=ff_mul, name='encoder_block_%d' % i) for i in range(blocks)]

        self.dropout = Dropout(dropout)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        # Map down to internal dimensionality if input isn't already in it.
        if not self.hidden_size == x.shape[-1]:
            x = self.projection(x)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        x = self.dropout(x, training=training)
        return x
