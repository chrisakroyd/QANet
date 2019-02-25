import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, SeparableConv1D
from src import layers


class EncoderBlock(tf.keras.Model):
    def __init__(self, conv_layers, kernel_size, block_number=0, total_blocks=1,
                 hidden_size=128, heads=8, dropout=0.1, attn_dropout=0.1, ff_inner_size=1.0, **kwargs):
        """ Builds an encoder block.

            Encoder block from the paper "QANet" (https://arxiv.org/pdf/1804.09541.pdf, section 2.2), it is roughly
            equivalent to a transformer block from "Attention is all you Need" (https://arxiv.org/pdf/1706.03762.pdf,
            section 3), the main differences are the separable convolutions before the self-attention layer and
            Layer Dropout.

            Each block after the position encoding is wrapped with a SublayerConnection which implments
            common functionality for the residual connection, layer norm and dropout. This gives each 'block' the
            structure input -> LayerNorm -> WrappedLayer -> Dropout -> residual. It also implements the
            regularization method layer dropout from the paper "Deep Networks with Stochastic Depth"
            (https://arxiv.org/pdf/1603.09382.pdf).

            Args:
                conv_layers: Number of convolutional layers in this block.
                kernel_size: Width of the kernel in this block.
                block_number: This blocks position within the stack.
                total_blocks: Total blocks within the stack.
                hidden_size: Number of filters in each block.
                heads: Number of attention heads to use.
                dropout: Fraction of input units to drop in all dropout layers within this block.
                ff_inner_size: Number of units in the inner non-linear layer of the feed-forward block.
        """
        super(EncoderBlock, self).__init__(**kwargs)
        # These Ids and counts are for determining layer dropout, higher layers == more chance of dropout
        self.block_start_id = (block_number * (conv_layers + 2)) + 1  # Start from one
        self.self_attention_id = (self.block_start_id + conv_layers)
        self.feed_forward_id = self.self_attention_id + 1
        self.total_sub_layers = (conv_layers + 2) * total_blocks

        # Block has 4 components, position encoding, n conv layers, a self-attention layer and a feed forward layer.
        # Can have n convs, create a list of conv blocks to iterate through incrementing their block_ids.
        self.conv_layers = [layers.SublayerConnection(SeparableConv1D(filters=hidden_size,
                                                                      kernel_size=kernel_size,
                                                                      padding='same',
                                                                      kernel_initializer=layers.create_initializer(),
                                                                      activation='relu'),
                                                      dropout=dropout,
                                                      sublayer=(self.block_start_id + i),
                                                      total_sublayers=self.total_sub_layers,
                                                      name='conv_block_%d' % (self.block_start_id + i))
                            for i in range(conv_layers)]

        self.self_attention = layers.SublayerConnection(layers.MultiHeadAttention(hidden_size,
                                                                                  num_heads=heads,
                                                                                  dropout=attn_dropout,
                                                                                  self_attention=True),
                                                        sublayer=self.self_attention_id,
                                                        total_sublayers=self.total_sub_layers,
                                                        dropout=dropout,
                                                        name='self_attention_%d' % self.self_attention_id)

        self.feed_forward = layers.SublayerConnection(layers.FeedForwardLayer(hidden_size, inner_size=ff_inner_size,
                                                                              dropout=dropout),
                                                      sublayer=self.feed_forward_id,
                                                      total_sublayers=self.total_sub_layers,
                                                      dropout=dropout,
                                                      name='feed_forward_%d' % self.feed_forward_id)

        self.output_normalization = layers.LayerNorm()

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single float32 tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        for conv in self.conv_layers:
            x = conv(x, training=training, mask=mask)

        x = self.self_attention(x, training=training, mask=mask)
        x = self.feed_forward(x, training=training, mask=mask)
        x = self.output_normalization(x)

        return x


class EncoderBlockStack(tf.keras.Model):
    def __init__(self, blocks, conv_layers, kernel_size, hidden_size=128, heads=8, dropout=0.1, attn_dropout=0.1, ff_inner_size=128,
                 **kwargs):
        """ Builds a stack of encoder blocks and handles input projection + output dropout.

            Wrapper around EncoderBlock that includes functionality for optional input projection,
            output dropout and creating multiple blocks with the same parameters.

            Args:
                blocks: Number of blocks in this stack.
                conv_layers: Number of convolutional layers in each block.
                kernel_size: Width of the kernels in each block.
                hidden_size: Number of filters in each block.
                heads: Number of attention heads to use.
                dropout: Fraction of input units to drop in all dropout layers within this stack.
                ff_inner_size: Number of units in the inner non-linear layer of the feed-forward block.
        """
        super(EncoderBlockStack, self).__init__(**kwargs)
        self.hidden_size = hidden_size

        self.projection = Conv1D(hidden_size,
                                 kernel_size=1,
                                 padding='same')

        self.position_encoding = layers.PositionEncoding()

        self.blocks = [EncoderBlock(conv_layers=conv_layers, kernel_size=kernel_size,
                                    hidden_size=hidden_size, heads=heads,
                                    dropout=dropout, attn_dropout=attn_dropout, block_number=i, total_blocks=blocks,
                                    ff_inner_size=ff_inner_size, name='encoder_block_%d' % i) for i in range(blocks)]

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

        x = self.position_encoding(x, training=training, mask=mask)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        x = self.dropout(x, training=training)
        return x
