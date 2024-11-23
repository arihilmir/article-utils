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
    kernel_size: Sequence[int] = (5,)

    def setup(self) -> None:
        self.additional_layers = [nn.RNN(
            CNNLSTMCell(features=self.hidden_size,
                        kernel_size=self.kernel_size, padding='same')
        )
            for _ in range(self.layers)]

    def __call__(self, xs: jax.Array):
        for l in self.additional_layers:
            xs = l(xs)

        return xs


class CNNLSTMCell(nn.RNNCellBase):
    features: int
    kernel_size: tp.Sequence[int]
    padding: int
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    param_dtype = jnp.float32

    def setup(self):
        conv_i = partial(
            nn.Conv,
            features=self.features,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        conv_h = partial(
            nn.Conv,
            features=self.features,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=False,
        )

        self.ii = conv_i()
        self.if_ = conv_i()
        self.ig = conv_i()
        self.io = conv_i()
        self.hi = conv_h()
        self.hf = conv_h()
        self.hg = conv_h()
        self.ho = conv_h()

    def __call__(self, carry: tuple[jax.Array, jax.Array], inputs: jax.Array) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        c, h = carry
        i = self.gate_fn(self.ii(inputs) + self.hi(h))
        f = self.gate_fn(self.if_(inputs) + self.hf(h))
        g = self.activation_fn(self.ig(inputs) + self.hg(h))
        o = self.gate_fn(self.io(inputs) + self.ho(h))

        fc = f * c
        ig = i * g
        new_c = fc + ig
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(
        self, rng: flax.typing.PRNGKey, input_shape: tuple[int, ...]
    ):
        batch_dims = input_shape[0]
        c = jnp.zeros((batch_dims, self.features), self.param_dtype)
        h = jnp.zeros((batch_dims, self.features), self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1

# MARK: - CNNGRU


class CNNGRUNet(nn.Module):
    hidden_size: int = 150
    layers: int = 1
    kernel_size: Sequence[int] = (5,)

    def setup(self) -> None:
        self.additional_layers = [nn.RNN(CNNGRUCell(features=self.hidden_size, kernel_size=self.kernel_size))
                                  for _ in range(self.layers)]

    def __call__(self, xs: jax.Array):
        for l in self.additional_layers:
            xs = l(xs)

        return xs


class CNNGRUCell(nn.Module):
    features: int = 150
    kernel_size: int = (5,)
    strides: int = 1
    padding: str = 'same'
    use_bias: bool = False
    dtype = None
    param_dtype = jnp.float32

    def setup(self):
        self.dense_i = nn.Conv(
            features=3*self.features,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=True,
            param_dtype=self.param_dtype,
        )

        self.dense_h = nn.Conv(
            features=3*self.features,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=False,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        carry: tuple[jax.Array, jax.Array],
        inputs: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        h = carry
        x_transformed = self.dense_i(inputs)
        h_transformed = self.dense_h(h)
        xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
        hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)

        # Compute gates
        r = nn.sigmoid(xi_r + hh_r)
        z = nn.sigmoid(xi_z + hh_z)

        # Compute n with an additional linear transformation on h
        n = nn.tanh(xi_n + r * hh_n)

        # Update hidden state
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    def initialize_carry(
        self, rng: flax.typing.PRNGKey, input_shape: tuple[int, ...]
    ):
        batch_dims = input_shape[0]
        return jnp.zeros((batch_dims, self.features), self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1

# MARK: - CNNRNN

classes = {'gru': GRU, 'lstm': LSTM}

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
                            self.layers, False)(x)

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
    hidden_size: int = 150
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

