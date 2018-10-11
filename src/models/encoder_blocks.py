import tensorflow as tf
import math
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Layer, SeparableConv1D
from src.models.utils import split_last_dimension, combine_last_two_dimensions, mask_logits
from src.models.layers import LayerDropout, LayerNorm


class PositionEncoding(Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def compute_input_shape(self, x):
        shape = tf.shape(x)
        return shape[1], shape[2]

    def call(self, x, training=None, mask=None):
        length, channels = self.compute_input_shape(x)
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        # Generate the signal with cos + sin waves.
        log_timescale_increment = (
                math.log(float(self.max_timescale) / float(self.min_timescale)) / (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # Add input and signal
        return x + signal

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, filters=128, num_heads=8, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.filters = filters

        self.memory_conv = Conv1D(2 * self.filters,
                                  kernel_size=1,
                                  strides=1,
                                  name='memory_projection',
                                  use_bias=False)

        self.query_conv = Conv1D(self.filters,
                                 kernel_size=1,
                                 strides=1,
                                 name='query_projection',
                                 use_bias=False)

        # square root of key depth https://arxiv.org/pdf/1706.03762.pdf (Attention is all you Need)
        self.scaling_factor = (self.filters // self.num_heads) ** -0.5

        self.softmax = Activation('softmax')

        self.dropout = Dropout(dropout)

    def call(self, queries, training=None, mask=None):
        memory = queries
        memory = self.memory_conv(memory)
        query = self.query_conv(queries)

        Q = split_last_dimension(query, self.num_heads)
        K, V = [split_last_dimension(tensor, self.num_heads) for tensor in tf.split(memory, 2, axis=2)]

        # @TODO From Attention is all you need, scaling factor should apply to the result of matmul op(logits)
        Q *= self.scaling_factor

        logits = tf.matmul(Q, K, transpose_b=True)

        if mask is not None:
            mask = tf.reshape(mask, shape=[tf.shape(logits)[0], 1, 1, -1])
            logits = mask_logits(logits, mask)

        weights = self.softmax(logits)
        weights = self.dropout(weights, training=training)
        x = tf.matmul(weights, V)

        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


class StackedConvBlock(tf.keras.Sequential):
    def __init__(self, filters, kernel_size, conv_layers, block_start_id, total_sub_layers, dropout=0.1, **kwargs):
        super(StackedConvBlock, self).__init__(**kwargs)
        for i in range(conv_layers):
            self.add(ConvBlock(filters=filters,
                               kernel_size=kernel_size,
                               dropout=dropout,
                               sub_layer_id=(block_start_id + 1) + i,
                               total_sub_layers=total_sub_layers))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training, mask=mask)
        return x


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, dropout, sub_layer_id, total_sub_layers, **kwargs):
        super(ConvBlock, self).__init__(name='conv_block_%d' % sub_layer_id, **kwargs)
        self.sub_layer_id = sub_layer_id
        self.total_sub_layers = total_sub_layers

        self.seperable_conv = SeparableConv1D(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=1,
                                              padding='same',
                                              use_bias=True,
                                              activation='relu')

        self.layer_norm = LayerNorm()

        self.dropout = Dropout(dropout)
        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x

        x = self.layer_norm(x)

        # Only apply dropout every 2 positions.
        if self.sub_layer_id % 2 == 0:
            x = self.dropout(x, training=training)

        x = self.seperable_conv(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class SelfAttentionBlock(tf.keras.Model):
    def __init__(self, filters, sub_layer_id, total_sub_layers, heads=8, dropout=0.0, **kwargs):
        super(SelfAttentionBlock, self).__init__(name='self_attention_%d' % sub_layer_id, **kwargs)
        self.sub_layer_id = sub_layer_id
        self.total_sub_layers = total_sub_layers
        self.layer_norm = LayerNorm()
        self.dropout = Dropout(dropout)
        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

        self.multi_head_attention = MultiHeadAttention(filters,
                                                       num_heads=heads,
                                                       dropout=dropout,
                                                       name='multi_head_attention')

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = self.multi_head_attention(x, training=training, mask=mask)
        x = self.layer_dropout([x, residual], training=training)
        return x


class FeedForwardBlock(tf.keras.Model):
    def __init__(self, filters, dropout, sub_layer_id, total_sub_layers, **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.layer_norm = LayerNorm()
        self.sub_layer_id = sub_layer_id
        self.total_sub_layers = total_sub_layers
        self.dropout = Dropout(dropout)
        # Feed forward layers, follows Attention is all you need. (Position-wise Feed-Forward Networks)
        self.conv_ff_1 = Conv1D(filters,
                                kernel_size=1,
                                strides=1,
                                use_bias=True,
                                name='conv_ff_1',
                                activation='relu')

        self.conv_ff_2 = Conv1D(filters,
                                kernel_size=1,
                                strides=1,
                                use_bias=True,
                                name='conv_ff_2')

        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = self.conv_ff_1(x)
        x = self.conv_ff_2(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class EncoderBlock(tf.keras.Model):
    def __init__(self, conv_layers, kernel_size, block_number=0, total_blocks=1,
                 filters=128, heads=8, dropout=0.1, **kwargs):
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
                                                       dropout=dropout)

        self.feed_forward_block = FeedForwardBlock(filters, dropout,
                                                   sub_layer_id=self.feed_forward_id,
                                                   total_sub_layers=self.total_sub_layers)

    def call(self, x, training=None, mask=None):
        x = self.position_encoding(x, training=training, mask=mask)
        x = self.stacked_conv_blocks(x, training=training, mask=mask)
        x = self.self_attention_block(x, training=training, mask=mask)
        x = self.feed_forward_block(x, training=training, mask=mask)
        return x


class StackedEncoderBlocks(tf.keras.Sequential):
    def __init__(self, blocks, conv_layers, kernel_size, filters=128, heads=8, dropout=0.1, **kwargs):
        super(StackedEncoderBlocks, self).__init__(**kwargs)

        for i in range(blocks):
            self.add(
                EncoderBlock(conv_layers=conv_layers, kernel_size=kernel_size,
                             filters=filters, heads=heads,
                             dropout=dropout, block_number=i, total_blocks=blocks,
                             name='encoder_block_%d' % i)
            )

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training, mask=mask)
        return x
