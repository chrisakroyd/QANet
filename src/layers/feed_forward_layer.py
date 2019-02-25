from tensorflow.keras.layers import Conv1D, Dropout, Layer
from src import layers


class FeedForwardLayer(Layer):
    def __init__(self, hidden_size=128, inner_size=128, dropout=0.1, **kwargs):
        """ Feed forward block

            Follows paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf, section 3.3) with
            optional parameter for increasing dimension of non-linear layer.

            Args:
                hidden_size: Number of hidden units.
                dropout: Fraction of input units to drop.
                inner_size: Number of units in the inner non-linear layer.
        """
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.filters = hidden_size
        # Optionally increase non-linear layer units by a multiplier
        self.conv_ff_1 = Conv1D(inner_size,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=layers.create_initializer(),
                                name='conv_ff_1',
                                activation='relu')

        self.dropout = Dropout(dropout)

        self.conv_ff_2 = Conv1D(hidden_size,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=layers.create_initializer(),
                                name='conv_ff_2')

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A tensor of shape [batch_size, seq_length, ?].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        x = self.conv_ff_1(x)
        x = self.dropout(x, training=training)
        x = self.conv_ff_2(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.filters
