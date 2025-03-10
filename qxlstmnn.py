import keras
from keras import ops
import tensorflow as tf


class QmLSTMCell(keras.layers.Layer):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.units = units
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]  # number of features
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='zeros',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='zeros',
            name="recurrent_kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name="bias",
            initializer='zeros',
        )
        super().build(input_shape)  # mark that model is built

    def call(self, inputs, state):
        h_state = state[0]
        c_state = state[1]

        z = ops.matmul(inputs, self.kernel)
        z += ops.matmul(h_state, self.recurrent_kernel)
        z += self.bias

        z0, z1, z2, z3 = ops.split(z, 4, axis=1)
        i = ops.sigmoid(z0)
        f = ops.sigmoid(z1)
        c = f * c_state + i * ops.sigmoid(z2)
        o = ops.sigmoid(z3)

        h = o*ops.tanh(c)
        return h, [h, c]

    def get_config(self):
        config = super().get_config()
        return config


class QmLSTM(keras.layers.RNN):
    def __init__(self, units, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, zero_output_for_mask=False, **kwargs):
        cell = QmLSTMCell(units)
        super().__init__(cell, return_sequences, return_state, go_backwards,
                         stateful, unroll, zero_output_for_mask, **kwargs)
