import tensorflow as tf
import math
from src.models.utils import split_last_dimension, combine_last_two_dimensions, mask_logits


class PositionEncoding(tf.keras.layers.Layer):
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


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-6

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1], ),
                                     initializer=tf.ones_initializer(),
                                     trainable=True,
                                     name='layer_norm_scale')

        self.bias = self.add_weight(shape=(input_shape[-1], ),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True,
                                    name='layer_norm_bias')
        self.built = True

    def call(self, x, training=None, mask=None):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm = (x - mean) * tf.rsqrt(variance + self.epsilon)
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerDropout(tf.keras.layers.Layer):
    def __init__(self, dropout, sublayer, total_sublayers, **kwargs):
        super(LayerDropout, self).__init__(**kwargs)
        self.Pl = dropout * float(sublayer) / float(total_sublayers)
        self.residual_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super(LayerDropout, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x, residual = x

        if training:
            pred = tf.random_uniform([]) < self.Pl
            # @TODO Should only use dropout every second layer.
            return tf.cond(pred, lambda: residual, lambda: self.residual_dropout(x) + residual)
        else:
            return x + residual

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, filters=128, num_heads=8, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(name='multi_head_attention', **kwargs)
        self.num_heads = num_heads
        self.filters = filters

        self.memory_conv = tf.keras.layers.Conv1D(2 * self.filters,
                                                  kernel_size=1,
                                                  strides=1,
                                                  name='memory_projection',
                                                  use_bias=False)

        self.query_conv = tf.keras.layers.Conv1D(self.filters,
                                                 kernel_size=1,
                                                 strides=1,
                                                 name='query_projection',
                                                 use_bias=False)

        # square root of key depth https://arxiv.org/pdf/1706.03762.pdf (Attention is all you Need)
        self.scaling_factor = (self.filters // self.num_heads) ** -0.5

        self.softmax = tf.keras.layers.Activation('softmax')

        self.dropout = tf.keras.layers.Dropout(dropout)

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
        weights = self.dropout(weights)
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
        self.dropout_val = dropout

        self.seperable_conv = tf.keras.layers.SeparableConv1D(filters=filters,
                                                              kernel_size=kernel_size,
                                                              strides=1,
                                                              padding='same',
                                                              use_bias=True,
                                                              activation='relu')

        self.layer_norm = LayerNorm()

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x

        x = self.layer_norm(x)

        # Only apply dropout every 2 positions.
        if self.sub_layer_id % 2 == 0:
            x = self.dropout(x)

        x = self.seperable_conv(x)
        x = self.layer_dropout([x, residual], training=training)
        return x


class SelfAttentionBlock(tf.keras.Model):
    def __init__(self, filters, sub_layer_id, total_sub_layers, heads=8, dropout=0.0, **kwargs):
        super(SelfAttentionBlock, self).__init__(name='self_attention_%d' % sub_layer_id, **kwargs)
        self.sub_layer_id = sub_layer_id
        self.total_sub_layers = total_sub_layers
        self.dropout_val = dropout
        self.layer_norm = LayerNorm()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

        self.multi_head_attention = MultiHeadAttention(filters,
                                                       num_heads=heads,
                                                       dropout=dropout)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.multi_head_attention(x, training=training, mask=mask)
        x = self.layer_dropout([x, residual], training=training)
        return x


class FeedForwardBlock(tf.keras.Model):
    def __init__(self, filters, dropout, sub_layer_id, total_sub_layers, **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.dropout_val = dropout
        self.layer_norm = LayerNorm()
        self.sub_layer_id = sub_layer_id
        self.total_sub_layers = total_sub_layers
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Our convoloutional feed forward layers
        self.conv_ff_1 = tf.keras.layers.Conv1D(filters,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=True,
                                                name='conv_ff_1',
                                                activation='relu')

        self.conv_ff_2 = tf.keras.layers.Conv1D(filters,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=True,
                                                name='conv_ff_2')

        self.layer_dropout = LayerDropout(dropout, sub_layer_id, total_sub_layers)

    def call(self, x, training=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.dropout(x)
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
