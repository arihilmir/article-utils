import keras
from keras import layers
from keras import ops
import numpy as np
import keras_tuner as kt
from functools import partial
from keras import backend as K


def standardize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.

    Args:
        value: int or iterable of ints. The value to validate and convert.
        n: int. The size of the tuple to be returned.
        name: string. The name of the argument being validated, e.g. "strides"
            or "kernel_size". This is only used to format error messages.
        allow_zero: bool, defaults to `False`. A `ValueError` will raised
            if zero is received and this argument is `False`.

    Returns:
        A tuple of n integers.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} integers. "
        f"Received {name}={value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f", including values {unqualified_values}"
            f" that do not satisfy `value {req_msg}`"
        )
        raise ValueError(error_msg)

    return value_tuple


def compute_conv_output_shape(
    input_shape,
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """Compute the output shape of conv ops."""
    if data_format == "channels_last":
        spatial_shape = input_shape[1:-1]
        kernel_shape = kernel_size + (input_shape[-1], filters)
    else:
        spatial_shape = input_shape[2:]
        kernel_shape = kernel_size + (input_shape[1], filters)
    if len(kernel_shape) != len(input_shape):
        raise ValueError(
            "Kernel shape must have the same length as input, but received "
            f"kernel of shape {kernel_shape} and "
            f"input of shape {input_shape}."
        )
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * len(spatial_shape)
    if len(dilation_rate) != len(spatial_shape):
        raise ValueError(
            "Dilation must be None, scalar or tuple/list of length of "
            "inputs' spatial shape, but received "
            f"`dilation_rate={dilation_rate}` and "
            f"input of shape {input_shape}."
        )
    none_dims = []
    spatial_shape = np.array(spatial_shape)
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            # Set `None` shape to a manual value so that we can run numpy
            # computation on `spatial_shape`.
            spatial_shape[i] = -1
            none_dims.append(i)

    kernel_spatial_shape = np.array(kernel_shape[:-2])
    dilation_rate = np.array(dilation_rate)
    if padding == "valid":
        output_spatial_shape = (
            np.floor(
                (spatial_shape - dilation_rate * (kernel_spatial_shape - 1) - 1)
                / strides
            )
            + 1
        )
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={input_shape}`, "
                    f"`kernel shape={kernel_shape}`, "
                    f"`dilation_rate={dilation_rate}`."
                )
    elif padding == "same" or padding == "causal":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    else:
        raise ValueError(
            "`padding` must be either `'valid'` or `'same'`. Received "
            f"{padding}."
        )
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
    if data_format == "channels_last":
        output_shape = (
            (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
        )
    else:
        output_shape = (
            input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape


def standardize_data_format(data_format):
    if data_format is None:
        return keras.backend.image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


class ConvGRUCell(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed=seed)
        self.rank = 1
        self.filters = filters
        self.kernel_size = standardize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        self.strides = standardize_tuple(
            strides, self.rank, "strides", allow_zero=True
        )
        self.padding = padding
        self.data_format = standardize_data_format(data_format)
        self.dilation_rate = standardize_tuple(
            dilation_rate, self.rank, "dilation_rate"
        )
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = keras.initializers.get(
            recurrent_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(
            recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = keras.constraints.get(recurrent_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.dropout_mask_count = 3
        self.input_spec = layers.InputSpec(ndim=self.rank + 2)
        self.state_size = -1  # Custom, defined in methods

    def build(self, inputs_shape, states_shape=None):
        if self.data_format == "channels_first":
            channel_axis = 1
            self.spatial_dims = inputs_shape[2:]
        else:
            channel_axis = -1
            self.spatial_dims = inputs_shape[1:-1]
        if None in self.spatial_dims:
            raise ValueError(
                "ConvGRU layers only support static input shapes for the spatial dimension."
                f" Received invalid input shape: input_shape={inputs_shape}"
            )
        if inputs_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs (last axis) should be defined. Found None. Full input shape received:"
                f" input_shape={inputs_shape}"
            )
        self.input_spec = layers.InputSpec(
            ndim=self.rank + 3, shape=(None,) + inputs_shape[1:]
        )

        input_dim = inputs_shape[channel_axis]
        self.input_dim = input_dim
        self.kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        recurrent_kernel_shape = self.kernel_size + (
            self.filters,
            self.filters * 3,
        )

        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters * 3,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=False):
        # Extract the previous hidden state
        # Expected shape: (batch_size, spatial_dims, filters)
        h_tm1 = states[0]

        print(self.kernel)
        # Compute inputs and hidden state projections in one step
        matrix_x = self.input_conv(
            inputs, self.kernel, self.bias, padding=self.padding)
        x_z, x_r, x_h = ops.split(matrix_x, 3, axis=-1)

        matrix_inner = self.recurrent_conv(h_tm1, self.recurrent_kernel)
        recurrent_z, recurrent_r, recurrent_h = ops.split(
            matrix_inner, 3, axis=-1)

        # Update and reset gates
        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # Candidate hidden state
        hh = self.activation(x_h + r * recurrent_h)

        # Compute the new hidden state
        h = z * h_tm1 + (1 - z) * hh

        # Return the new hidden state and updated states
        return h, [h]

    def compute_output_shape(self, inputs_shape, states_shape=None):
        conv_output_shape = compute_conv_output_shape(
            inputs_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        return conv_output_shape, [conv_output_shape]

    def get_initial_state(self, batch_size=None):
        if self.data_format == "channels_last":
            input_shape = (batch_size,) + self.spatial_dims + (self.input_dim,)
        else:
            input_shape = (batch_size, self.input_dim) + self.spatial_dims
        state_shape = self.compute_output_shape(input_shape)[0]
        return [ops.zeros(state_shape, dtype=self.compute_dtype)]

    def input_conv(self, x, w, b=None, padding="valid"):
        # Adjusted for rank-1 to handle 1D convolutions correctly
        conv_out = ops.conv(
            x,
            w,
            strides=self.strides,
            padding=padding,
            data_format="channels_last",  # Enforce channels_last for rank-1 inputs
            dilation_rate=self.dilation_rate,
        )
        if b is not None:
            bias_shape = (1, 1, self.filters * 3)  # Adjust bias shape for 1D
            b = ops.reshape(b, bias_shape)
            conv_out += b
        return conv_out

    def recurrent_conv(self, x, w):
        # Adjusted for rank-1 to support 1D convolutions
        strides = standardize_tuple(
            1, self.rank, "strides", allow_zero=True
        )
        conv_out = ops.conv(
            x, w, strides=strides, padding="same", data_format="channels_last"
        )
        return conv_out

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": keras.activations.serialize(self.activation),
            "recurrent_activation": keras.activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": keras.regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": keras.constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ConvGRU(layers.RNN):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        **kwargs,
    ):
        cell = ConvGRUCell(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            name="conv_gru_cell",
            dtype=kwargs.get("dtype"),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )
        self.input_spec = layers.InputSpec(ndim=cell.rank + 3)

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences, initial_state=initial_state, mask=mask, training=training
        )

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        batch_size = sequences_shape[0]
        steps = sequences_shape[1]
        step_shape = (batch_size,) + sequences_shape[2:]
        state_shape = self.cell.compute_output_shape(step_shape)[0][1:]

        if self.return_sequences:
            output_shape = (
                batch_size,
                steps,
            ) + state_shape
        else:
            output_shape = (batch_size,) + state_shape

        if self.return_state:
            batched_state_shape = (batch_size,) + state_shape
            return output_shape, batched_state_shape, batched_state_shape
        return output_shape

    def compute_mask(self, _, mask):
        mask = keras.tree.flatten(mask)[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None, None]
            return [output_mask] + state_mask
        else:
            return output_mask

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": keras.activations.serialize(self.activation),
            "recurrent_activation": keras.activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": keras.regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": keras.constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
