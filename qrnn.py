import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np


def _dropout(x, level, noise_shape=None, seed=None):
    # Custom dropout that compensates for the default scaling
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level)  # undo the scaling by (1 - p)
    return x

def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Copy of the function of keras-team/keras because it's not in the public API
    So we can't use the function in keras-team/keras to test tf.keras

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

class QRNN(Layer):
    """
    Quasi RNN (QRNN) Layer

    # Arguments
        units: dimension of the internal projections and the final output.
        window_size: Size of the convolution window.
        stride: Stride for the convolution.
        return_sequences: Whether to return the full sequence or just the last output.
        go_backwards: Process the input sequence backwards.
        stateful: Whether this layer should maintain a persistent state across batches.
        unroll: Whether to unroll the QRNN.
        activation: Activation function (default is 'tanh').
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias.
        activity_regularizer: Regularizer function applied to the output of the layer.
        kernel_constraint: Constraint function for kernel weights.
        bias_constraint: Constraint function for bias.
        dropout: Float between 0 and 1. Fraction of the units to drop.
        use_bias: Whether to use a bias vector.
        input_dim: Dimensionality of the input (for sequential models).
        input_length: Length of the input sequence (for sequential models).

    # References:
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    """

    def __init__(
        self,
        units,
        window_size=2,
        stride=1,
        return_sequences=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        activation='tanh',
        kernel_initializer='uniform',
        bias_initializer='zero',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dropout=0,
        use_bias=True,
        input_dim=None,
        input_length=None,
        **kwargs
    ):
        super(QRNN, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = dropout
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length

        # For Sequential models that specify input_length/input_dim:
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))
        self.states = [None]

        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units * 3,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            # In tf.keras, use conv_utils.conv_output_length
            length = conv_output_length(
                length + self.window_size - 1,
                self.window_size,
                padding='valid',
                stride=self.strides[0],
            )
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        return [initial_state for _ in range(len(self.states))]

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError(
                'If a QRNN is stateful, it needs to know its batch size. '
                'Specify the batch size of your input tensors.'
            )

        if self.states[0] is None:
            self.states = [
                K.zeros((batch_size, self.units)) for _ in self.states
            ]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError(
                    f'Layer {self.name} expects {len(self.states)} states, '
                    f'but it received {len(states)} state values.'
                )
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError(
                        f'State {index} is incompatible with layer {self.name}: '
                        f'expected shape={(batch_size, self.units)}, found shape={value.shape}'
                    )
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Combine inputs + initial_state into a single list
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            initial_states = [initial_state]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError(
                f'Layer has {len(self.states)} states but was passed '
                f'{len(initial_states)} initial states.'
            )

        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError(
                'Cannot unroll a RNN if the time dimension is undefined.'
            )

        constants = self.get_constants(inputs, training=training)
        preprocessed_input = self.preprocess_input(inputs, training=training)

        last_output, outputs, states = K.rnn(
            self.step,
            preprocessed_input,
            initial_states,
            go_backwards=self.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.unroll,
            input_length=input_shape[1],
        )

        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # If dropout > 0, some outputs might depend on learning_phase
        if 0 < self.dropout < 1:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        # Padding to the left so we can do a "valid" conv across timesteps
        if self.window_size > 1:
            inputs = tf.keras.backend.temporal_padding(inputs, (self.window_size - 1, 0))

        # Convolution requires a 4D tensor
        inputs = K.expand_dims(inputs, 2)  # (batch, time, 1, input_dim)

        output = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding='valid',
            data_format='channels_last',
        )
        output = K.squeeze(output, 2)  # remove the dummy dimension -> (batch, new_time, units*3)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        # If dropout is set, apply the custom dropout to the "forget gate" portion
        if self.dropout and 0.0 < self.dropout < 1.0:
            z = output[:, :, : self.units]
            f = output[:, :, self.units : 2 * self.units]
            o = output[:, :, 2 * self.units :]

            # Only in training, transform f -> 1 - dropout(1 - f)
            # This mimics 'zoneout' style gating used in some QRNN implementations
            f = K.in_train_phase(
                1 - _dropout(1 - f, self.dropout),  # if training
                f,                                 # if not training
                training=training,
            )
            return K.concatenate([z, f, o], axis=-1)
        else:
            return output

    def step(self, inputs, states):
        prev_output = states[0]

        z = inputs[:, : self.units]
        f = inputs[:, self.units : 2 * self.units]
        o = inputs[:, 2 * self.units :]

        z = self.activation(z)
        if self.dropout and 0.0 < self.dropout < 1.0:
            # We already applied partial "zoneout" to f in preprocess_input,
            # so just pass it here:
            pass
        else:
            f = K.sigmoid(f)
        o = K.sigmoid(o)

        output = f * prev_output + (1 - f) * z
        output = o * output

        return output, [output]

    def get_constants(self, inputs, training=None):
        return []

    def get_config(self):
        config = {
            'units': self.units,
            'window_size': self.window_size,
            'stride': self.strides[0],
            'return_sequences': self.return_sequences,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'unroll': self.unroll,
            'use_bias': self.use_bias,
            'dropout': self.dropout,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'input_dim': self.input_dim,
            'input_length': self.input_length,
        }
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

