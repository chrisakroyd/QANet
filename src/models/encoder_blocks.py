import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, SeparableConv1D
from src.models.layers import LayerDropout, LayerNorm, MultiHeadAttention, PositionEncoding


class StackedConvBlock(tf.keras.Sequential):
    def __init__(self, filters, kernel_size, conv_layers, block_start_id, total_sub_layers, dropout=0.1, **kwargs):
        super(StackedConvBlock, self).__init__(**kwargs)
        for i in range(conv_layers):
            sub_layer_id = block_start_id + i
            self.add(ConvBlock(filters=filters,
                               kernel_size=kernel_size,
                               dropout=dropout,
                               sub_layer_id=sub_layer_id,
                               total_sub_layers=total_sub_layers,
                               name='conv_block_%d' % sub_layer_id))

    def call(self, x, training=None, mask=None):
        for layer in self.layers:
            x = layer(x, training=training, mask=mask)
        return x


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, dropout, sub_layer_id, total_sub_layers, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_norm = LayerNorm()

        self.seperable_conv = SeparableConv1D(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=1,
                                              padding='same',
                                              use_bias=True,
                                              activation='relu')

        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.seperable_conv(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class SelfAttentionBlock(tf.keras.Model):
    def __init__(self, filters, sub_layer_id, total_sub_layers, heads=8, dropout=0.0, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.layer_norm = LayerNorm()
        self.multi_head_attention = MultiHeadAttention(filters,
                                                       num_heads=heads,
                                                       dropout=dropout,
                                                       name='multi_head_attention')
        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.multi_head_attention([x, x], training=training, mask=mask)
        x = self.layer_dropout([x, residual], training=training)
        return x


class FeedForwardBlock(tf.keras.Model):
    def __init__(self, filters, dropout, sub_layer_id, total_sub_layers, ff_mul=1.0, **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.layer_norm = LayerNorm()
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

        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

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
        self.position_encoding = PositionEncoding()
        # Can have n convs so just give the start id for this block.
        self.stacked_conv_blocks = StackedConvBlock(filters=filters,
                                                    kernel_size=kernel_size,
                                                    conv_layers=conv_layers,
                                                    dropout=dropout,
                                                    block_start_id=self.block_start_id,
                                                    total_sub_layers=self.total_sub_layers,
                                                    name='%d_conv_blocks' % conv_layers)

        # Just give them a sub layer id that represents their position
        self.self_attention_block = SelfAttentionBlock(filters,
                                                       heads=heads,
                                                       sub_layer_id=self.self_attention_id,
                                                       total_sub_layers=self.total_sub_layers,
                                                       dropout=dropout,
                                                       name='self_attention_%d' % self.self_attention_id)

        self.feed_forward_block = FeedForwardBlock(filters, dropout,
                                                   ff_mul=ff_mul,
                                                   sub_layer_id=self.feed_forward_id,
                                                   total_sub_layers=self.total_sub_layers,
                                                   name='feed_forward_%d' % self.feed_forward_id)

    def call(self, x, training=None, mask=None):
        x = self.position_encoding(x, training=training, mask=mask)
        x = self.stacked_conv_blocks(x, training=training, mask=mask)
        x = self.self_attention_block(x, training=training, mask=mask)
        x = self.feed_forward_block(x, training=training, mask=mask)
        return x


class StackedEncoderBlocks(tf.keras.Sequential):
    def __init__(self, blocks, conv_layers, kernel_size, filters=128, heads=8, dropout=0.1, ff_mul=1.0, **kwargs):
        super(StackedEncoderBlocks, self).__init__(**kwargs)
        for i in range(blocks):
            self.add(
                EncoderBlock(conv_layers=conv_layers, kernel_size=kernel_size,
                             filters=filters, heads=heads,
                             dropout=dropout, block_number=i, total_blocks=blocks,
                             ff_mul=ff_mul, name='encoder_block_%d' % i)
            )

    def call(self, x, training=None, mask=None):
        for layer in self.layers:
            x = layer(x, training=training, mask=mask)
        return x
