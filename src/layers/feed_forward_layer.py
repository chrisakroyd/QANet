from tensorflow.keras.layers import Conv1D, Dropout, Layer


class FeedForwardLayer(Layer):
    def __init__(self, filters=128, ff_mul=1.0, dropout=0.1, **kwargs):
        """ Feed forward block

            Follows paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf, section 3.3) with
            optional parameter for increasing dimension of non-linear layer.

            Args:
                filters: Number of filters in each block.
                dropout: Fraction of input units to drop.
                ff_mul: Feed-Forward Multiplier, increases the number of units in the first feed-forward
                        layer by a float multiplier.
        """
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.filters = filters
        # Optionally increase non-linear layer units by a multiplier
        self.conv_ff_1 = Conv1D(int(filters * ff_mul),
                                kernel_size=1,
                                padding='same',
                                name='conv_ff_1',
                                activation='relu')

        self.dropout = Dropout(dropout)

        self.conv_ff_2 = Conv1D(filters,
                                kernel_size=1,
                                padding='same',
                                name='conv_ff_2')

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A tensor of shape [batch_size, seq_length, ?].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        x = self.conv_ff_1(x)
        x = self.dropout(x)
        x = self.conv_ff_2(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.filters
