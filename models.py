from typing import Any, Callable, Sequence
from flax import linen as nn
from functools import partial
import flax
import flax.typing
import jax
import jax.numpy as jnp
import typing as tp


@jax.jit
def silu(array: jax.Array) -> jax.Array:
    return array * nn.sigmoid(array)

# MARK: - LSTM


class LSTM(nn.Module):
    hidden_size: int = 150
    layers: int = 1

    def setup(self) -> None:
        self.lstm_net = nn.RNN(
            nn.LSTMCell(features=self.hidden_size),
        )
        self.additional_layers = [nn.RNN(nn.LSTMCell(features=self.hidden_size))
                                  for _ in range(self.layers-1)]

    def __call__(self, xs: jax.Array):
        xs = self.lstm_net(xs)
        for l in self.additional_layers:
            xs = l(xs)

        return xs

# MARK: GRU


class GRU(nn.Module):
    hidden_size: int = 150
    layers: int = 1

    def setup(self) -> None:
        self.additional_layers = [nn.RNN(nn.GRUCell(features=self.hidden_size))
                                  for _ in range(self.layers)]

    def __call__(self, xs: jax.Array):
        for l in self.additional_layers:
            xs = l(xs)

        return xs


# MARK: - CNNLSTM

class CNNLSTM(nn.Module):
    hidden_size: int = 150
    layers: int = 1
    kernel_size: Sequence[int] = (4,)

    def setup(self) -> None:
        self.additional_layers = [nn.RNN(
            nn.ConvLSTMCell(features=self.hidden_size,
                            kernel_size=self.kernel_size)
        )
            for _ in range(self.layers)]

    def __call__(self, xs: jax.Array):
        for l in self.additional_layers:
            xs = l(xs)

        return xs


class CNNGRUNet(nn.Module):
    hidden_size: int = 150
    layers: int = 1
    kernel_size: Sequence[int] = (5,)

    def setup(self) -> None:
        self.additional_layers = [nn.RNN(
            CNNGRUCell(features=self.hidden_size,
                       kernel_size=self.kernel_size))
            for _ in range(self.layers)]

    def __call__(self, xs: jax.Array):
        for l in self.additional_layers:
            xs = l(xs)

        return xs


class CNNLSTMCell(nn.RNNCellBase):
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] | None = None
    padding: str | Sequence[tuple[int, int]] = 'SAME'
    use_bias: bool = True
    dtype: flax.typing.Dtype | None = None
    param_dtype: flax.typing.Dtype = jnp.float32
    carry_init: flax.typing.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, carry, inputs):
        """Constructs a convolutional LSTM.

        Args:
          carry: the hidden state of the Conv2DLSTM cell,
            initialized using ``Conv2DLSTM.initialize_carry``.
          inputs: input data with dimensions (batch, spatial_dims..., features).
        Returns:
          A tuple with the new carry and the output.
        """
        c, h = carry
        input_to_hidden = partial(
            nn.Conv,
            features=4 * self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='ih',
        )

        hidden_to_hidden = partial(
            nn.Conv,
            features=4 * self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='hh',
        )

        gates = input_to_hidden()(inputs) + hidden_to_hidden()(h)
        i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        f = nn.sigmoid(f)
        new_c = f * c + nn.sigmoid(i) * jnp.tanh(g)
        new_h = nn.sigmoid(o) * jnp.tanh(new_c)
        return (new_c, new_h), new_h

    @nn.nowrap
    def initialize_carry(self, rng: flax.typing.PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        # (*batch_dims, *signal_dims, features)
        signal_dims = input_shape[-self.num_feature_axes: -1]
        batch_dims = input_shape[: -self.num_feature_axes]
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + signal_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return c, h

    @property
    def num_feature_axes(self) -> int:
        return len(self.kernel_size) + 1

# MARK: - CNNGRU


class CNNGRUCell(nn.RNNCellBase):
    features: int = 150
    kernel_size: int = (5,)
    strides: int = 1
    padding: str = 'same'
    use_bias: bool = False
    dtype = None
    param_dtype = jnp.float32
    carry_init: flax.typing.Initializer = nn.initializers.zeros_init()

    def setup(self):
        hidden_features = self.features

        # Convolution layers for gates (input and hidden state)
        self.gates_i_conv = nn.Conv(
            features=3 * hidden_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='gates_i_conv'
        )
        self.gates_h_conv = nn.Conv(
            features=3 * hidden_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='gates_h_conv'
        )

    def __call__(self, carry, inputs):
        h = carry

        # Compute gates
        gates_i = self.gates_i_conv(inputs)
        gates_h = self.gates_h_conv(h)
        iir, iiz, iin = jnp.split(gates_i, 3, axis=-1)
        hir, hiz, hin = jnp.split(gates_h, 3, axis=-1)
        r = nn.sigmoid(iir + hir)
        z = nn.sigmoid(iiz + hiz)

        # Compute candidate hidden state
        n = nn.tanh(iin + r * hin)

        # Compute new hidden state
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    def initialize_carry(
        self, rng: flax.typing.PRNGKey, input_shape: tuple[int, ...]
    ):
        signal_dims = input_shape[-self.num_feature_axes: -1]
        batch_dims = input_shape[: -self.num_feature_axes]
        _, key2 = jax.random.split(rng)
        mem_shape = batch_dims + signal_dims + (self.features,)
        return self.carry_init(key2, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return len(self.kernel_size) + 1

# MARK: - CNNRNN


classes = {'gru': GRU, 'lstm': LSTM, 'cnngru': CNNGRUNet, 'cnnlstm': CNNLSTM}


class GlobalNet(nn.Module):
    rnn_cls: str = 'cnngru'
    hidden_size: int = 150
    out_features: int = 24
    kernel_size: Sequence[int] = (5,)
    layers: int = 1

    @nn.compact
    def __call__(self, xs: jax.Array):
        for i in range(self.layers):
            gru = classes[self.rnn_cls](self.hidden_size, self.kernel_size)
            carry = self.variable(
                'state', f'carry{i}', gru.initialize_carry, jax.random.PRNGKey(
                    0), xs.shape
            )
            new_c, out = gru(carry.value, xs)
            carry.value = new_c

        out = nn.Dense(self.hidden_size)(out[:, -1, :])
        out = silu(out)
        out = nn.Dense(self.out_features)(out)
        return jnp.expand_dims(out, axis=-1)


class CNNRNNNet(nn.Module):
    rnn_cls: str
    num_blocks: int = 1
    hidden_size: int = 150
    out_size: int = 24
    kernel_size: Sequence[int] = (3,)
    layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, train=True):
        for _ in range(self.num_blocks - 1):
            x = CNNRNNBlock(self.rnn_cls, self.hidden_size, self.hidden_size, self.kernel_size,
                            self.layers, True)(x)

        x = CNNRNNBlock(self.rnn_cls, self.hidden_size, self.out_size, self.kernel_size,
                        self.layers, True)(x)

        return x


class CNNRNNBlock(nn.Module):
    rnn_cls: str
    hidden_size: int
    out_size: int
    kernel_size: Sequence[int]
    layers: int = 1
    single_out: bool = False

    def setup(self) -> None:
        self.lstm = classes[self.rnn_cls](
            hidden_size=self.hidden_size, layers=self.layers, kernel_size=self.kernel_size)
        self.lin1 = nn.Dense(features=128)
        self.lin2 = nn.Dense(features=self.out_size)

    def __call__(self, x: jax.Array):
        x = self.lstm(x)
        if self.single_out:
            x = x[:, -1, :]
        x = self.lin1(x)
        x = silu(x)
        x = self.lin2(x)
        return jnp.expand_dims(x, axis=-1)

# MARK: - RNN


class RNNNet(nn.Module):
    rnn_cls: str
    num_blocks: int = 1
    hidden_size: Sequence[int] = (150)
    out_size: int = 24
    layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, train=True):
        for _ in range(self.num_blocks - 1):
            x = RNNBlock(self.rnn_cls, self.hidden_size, self.out_size,
                         self.layers, True)(x)

        x = RNNBlock(self.rnn_cls, self.hidden_size, self.out_size,
                     self.layers, True)(x)
        return x

class BatchNet(nn.Module):
    rnn_cls: str
    num_blocks: int = 1
    hidden_size: Sequence[int] = (150)
    out_size: int = 24
    layers: int = 1

    @nn.compact
    def __call__(self, xs, train=True):
        pc_xs, w_xs = jnp.split(xs, [1], axis=-1)
        pc_xs = RNNBlock(self.rnn_cls, hidden_size=self.hidden_size, out_size=self.hidden_size, single_out=True)(pc_xs)
        w_xs = RNNBlock(self.rnn_cls, hidden_size=self.hidden_size, out_size=self.hidden_size, single_out=True)(w_xs)

        xs = jnp.concatenate([pc_xs, w_xs], axis=-1)
        xs = RNNBlock(self.rnn_cls, hidden_size=self.hidden_size, out_size=self.out_size, single_out=True)(xs)
        return xs


class RNNBlock(nn.Module):
    rnn_cls: str
    hidden_size: int = 150
    out_size: int = 24
    layers: int = 1
    single_out: bool = False

    def setup(self) -> None:
        self.lstm = classes[self.rnn_cls](
            hidden_size=self.hidden_size, layers=self.layers)
        self.lin1 = nn.Dense(features=128)
        self.lin2 = nn.Dense(features=self.out_size)

    def __call__(self, x: jax.Array):
        x = self.lstm(x)
        if self.single_out:
            x = x[:, -1, :]
        x = self.lin1(x)
        x = silu(x)
        x = self.lin2(x)
        return jnp.expand_dims(x, axis=-1)
