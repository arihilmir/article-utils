from typing import Any
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
import optuna

from .trainer import TrainerModule
from .models import RNNBlock, RNNNet, GRU, LSTM, BatchNet



class RNNTrainer(TrainerModule):
    def __init__(
        self,
        rnn_cls: str,
        num_blocks: int = 1,
        hidden_size: int = 150,
        layers: int = 1,
        trial: Any = None,
        **kwargs
    ):
        super().__init__(
            model_class=BatchNet,
            model_hparams={
                "rnn_cls": rnn_cls,
                "num_blocks": num_blocks,
                "hidden_size": hidden_size,
                "layers": layers
            },
            **kwargs
        )
        self.trial = trial

    def create_functions(self):
        def calculate_loss(params, batch):
            inputs, targets = batch
            predictions = self.state.apply_fn({'params': params}, inputs)
            mae = jnp.abs((targets - predictions) / targets).mean()
            acc = jnp.sqrt(optax.squared_error(predictions, targets).mean())
            loss = optax.squared_error(predictions, targets).mean() # mse loss
            return loss, (acc, mae)

            # Training function
        def train_step(state, batch):
            def loss_fn(params): return calculate_loss(params, batch)
            ret, grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            loss, acc, mae = ret[0], ret[1][0], ret[1][1]
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss, 'acc': acc, 'mae': mae}
            return state, metrics

        # Evaluation function
        def eval_step(state, batch):
            loss, acc = calculate_loss(state.params, batch)
            return {'acc': acc[0], 'loss': loss, 'mae': acc[1]}

        return train_step, eval_step
