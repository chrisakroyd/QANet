from tensorflow.keras.layers import Conv1D, Dropout, Layer


class HighwayLayer(Layer):
    def __init__(self, dropout=0.1, use_bias=True, **kwargs):
        super(HighwayLayer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.gate_dropout = Dropout(dropout)
        self.trans_dropout = Dropout(dropout)

    def build(self, input_shape):
        filters = int(input_shape[-1])
        self.gate = Conv1D(filters,
                           kernel_size=1,
                           strides=1,
                           use_bias=self.use_bias,
                           padding='same',
                           activation='sigmoid',
                           name='gate')

        self.trans = Conv1D(filters,
                            kernel_size=1,
                            strides=1,
                            use_bias=self.use_bias,
                            padding='same',
                            activation='relu',
                            name='activation')
        super(HighwayLayer, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        gate_out = self.gate(x)
        gate_out = self.gate_dropout(gate_out, training=training)
        trans_out = self.trans(x)
        trans_out = self.trans_dropout(trans_out, training=training)
        out = gate_out * trans_out + (1 - gate_out) * x
        return out

    def compute_output_shape(self, input_shape):
        return input_shape
