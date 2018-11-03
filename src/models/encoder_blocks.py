import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, SeparableConv1D
from src import layers


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, dropout, sub_layer_id, total_sub_layers, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()

        self.seperable_conv = SeparableConv1D(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=1,
                                              padding='same',
                                              use_bias=True,
                                              activation='relu')

        self.layer_dropout = layers.LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.seperable_conv(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class SelfAttentionBlock(tf.keras.Model):
    def __init__(self, filters, sub_layer_id, total_sub_layers, heads=8, dropout=0.0, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()
        self.multi_head_attention = layers.MultiHeadAttention(filters,
                                                              num_heads=heads,
                                                              dropout=dropout,
                                                              name='multi_head_attention')
        self.layer_dropout = layers.LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.multi_head_attention([x, x], training=training, mask=mask)
        x = self.layer_dropout([x, residual], training=training)
        return x


class FeedForwardBlock(tf.keras.Model):
    def __init__(self, filters, dropout, sub_layer_id, total_sub_layers, ff_mul=1.0, **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()
        # Feed forward layers, follows Attention is all you need. (Position-wise Feed-Forward Networks),
        # optionally increase units in the first layer by a multiplier.
        self.conv_ff_1 = Conv1D(int(filters * ff_mul),
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                use_bias=True,
                                name='conv_ff_1',
                                activation='relu')
        # Attention is all you need has dropout after the relu conv, mirroring this here.
        self.dropout = Dropout(dropout)
        self.conv_ff_2 = Conv1D(filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                use_bias=True,
                                name='conv_ff_2')

        self.layer_dropout = layers.LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.conv_ff_1(x)
        x = self.dropout(x)
        x = self.conv_ff_2(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class EncoderBlock(tf.keras.Model):
    def __init__(self, conv_layers, kernel_size, block_number=0, total_blocks=1,
                 filters=128, heads=8, dropout=0.1, ff_mul=1.0, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        # These Ids and counts are for determining layer dropout, higher layers == more chance of dropout
        self.block_start_id = (block_number * (conv_layers + 2)) + 1  # Start from one
        self.self_attention_id = (self.block_start_id + conv_layers) + 1
        self.feed_forward_id = self.self_attention_id + 1
        self.total_sub_layers = (conv_layers + 2) * total_blocks

        # Each block has 4 components, Position encoding, n Conv Blocks, a self-attention block and a feed
        # forward block with residual connections between them (Implemented as part of the block)
        self.position_encoding = layers.PositionEncoding()
        # Can have n convs, create a list of conv blocks to iterate through incrementing their block_ids.
        self.conv_layers = [ConvBlock(filters=filters,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      sub_layer_id=(self.block_start_id + i),
                                      total_sub_layers=self.total_sub_layers,
                                      name='conv_block_%d' % (self.block_start_id + i)) for i in range(conv_layers)]

        self.self_attention = SelfAttentionBlock(filters,
                                                 heads=heads,
                                                 sub_layer_id=self.self_attention_id,
                                                 total_sub_layers=self.total_sub_layers,
                                                 dropout=dropout,
                                                 name='self_attention_%d' % self.self_attention_id)

        self.feed_forward = FeedForwardBlock(filters, dropout,
                                             ff_mul=ff_mul,
                                             sub_layer_id=self.feed_forward_id,
                                             total_sub_layers=self.total_sub_layers,
                                             name='feed_forward_%d' % self.feed_forward_id)

    def call(self, x, training=None, mask=None):
        x = self.position_encoding(x, training=training, mask=mask)
        for conv in self.conv_layers:
            x = conv(x, training=training, mask=mask)
        x = self.self_attention(x, training=training, mask=mask)
        x = self.feed_forward(x, training=training, mask=mask)
        return x


class StackedEncoderBlocks(tf.keras.Model):
    def __init__(self, blocks, conv_layers, kernel_size, filters=128, heads=8, dropout=0.1, ff_mul=1.0, **kwargs):
        super(StackedEncoderBlocks, self).__init__(**kwargs)
        self.hidden_size = filters
        # Apply a linear layer to map input dimensionality to internal dimensionality.
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
        # Map down to internal dimensionality if input isn't already in it.
        if not self.hidden_size == x.shape[-1]:
            x = self.projection(x)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        x = self.dropout(x, training=training)
        return x
