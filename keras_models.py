import keras
from keras import layers
from keras import ops
import keras_tuner as kt
from functools import partial
from keras import backend as K

class LSTMHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(72, 5)))

        num_units = hp.Choice('lstm_units', [32, 64])
        for _ in range(hp.Choice('num_layers', [0, 1])):
            model.add(layers.LSTM(num_units, return_sequences=True))
        model.add(layers.LSTM(num_units))

        model.add(layers.Dense(units=128, activation='silu'))
        model.add(layers.Dense(24))
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=[keras.metrics.MeanAbsolutePercentageError(),
                               keras.metrics.RootMeanSquaredError(),
                               keras.metrics.R2Score()])
        return model

class CNNLSTMHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(72, 5)))

        num_units = hp.Choice('lstm_units', [32, 64, 128])
        kernel_size = hp.Choice('kernel', [2, 4])
        for _ in range(hp.Choice('num_layers', [0, 1,])):
            model.add(ConvLSTM(num_units, kernel=(
                kernel_size,), return_sequences=True))
        model.add(ConvLSTM(num_units, kernel=(kernel_size,)))

        model.add(layers.Dense(units=128, activation='silu'))
        model.add(layers.Dense(24))
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=[keras.metrics.MeanAbsolutePercentageError(),
                               keras.metrics.RootMeanSquaredError(),
                               keras.metrics.R2Score()])
        return model

class ConvLSTMCell(layers.Layer):
    def __init__(self, units, kernel, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units]
        self.output_size = self.units

        cnv = partial(layers.Conv1D,
                      filters=units,
                      kernel_size=kernel,
                      data_format='channels_first',
                      activation='relu')

        self.input_gate = cnv()
        self.forget_gate = cnv()
        self.output_gate = cnv()
        self.cell_candidate_gate = cnv()

    def build(self, input_shape):
        input_dim = input_shape[-1]
        concat_dim = input_dim + self.units

        conv_input_shape = (None, 1, concat_dim)

        self.input_gate.build(conv_input_shape)
        self.forget_gate.build(conv_input_shape)
        self.output_gate.build(conv_input_shape)
        self.cell_candidate_gate.build(conv_input_shape)

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev, c_prev = states  # Previous hidden state and cell state

        # if h_prev.ndim == 2:
        h_prev = ops.expand_dims(h_prev, axis=1)
        inputs = ops.expand_dims(inputs, axis=1)

        concat_input = ops.concatenate([inputs, h_prev], axis=-1)

        # Compute gate values
        i = self.input_gate(concat_input)  # Input gate
        f = self.forget_gate(concat_input)  # Forget gate
        o = self.output_gate(concat_input)  # Output gate
        c_tilde = self.cell_candidate_gate(
            concat_input)  # Candidate cell state

        i = ops.squeeze(i, axis=1)
        f = ops.squeeze(f, axis=1)
        o = ops.squeeze(o, axis=1)
        c_tilde = ops.squeeze(c_tilde, axis=1)

        # Update cell state and hidden state
        c_new = f * c_prev + i * c_tilde
        h_new = o * ops.tanh(c_new)

        return h_new, [h_new, c_new]


class ConvLSTM(layers.RNN):
    def __init__(self, units, kernel, **kwargs):
        cell = ConvLSTMCell(units, kernel)
        super().__init__(cell, **kwargs)
