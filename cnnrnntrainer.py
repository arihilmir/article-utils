from typing import Any, Sequence
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
import optuna

from .trainer import TrainerModule
from .models import CNNRNNBlock, CNNRNNNet, GRU, LSTM


class GRUCNNRNNTrainer(TrainerModule):
    def __init__(
        self,
        rnn_cls: str,
        num_blocks: int = 1,
        hidden_size: int = 150,
        layers: int = 1,
        kernel_size: Sequence[int] = (3,),
        trial: Any = None,
        **kwargs
    ):
        super().__init__(
            model_class=CNNRNNNet,
            model_hparams={
                "rnn_cls": rnn_cls,
                "num_blocks": num_blocks,
                "hidden_size": hidden_size,
                "layers": layers,
                'kernel_size': kernel_size
            },
            **kwargs
        )
        self.trial = trial

    def create_functions(self):
        def calculate_loss(params, batch):
            inputs, targets = batch
            predictions = self.state.apply_fn({'params': params}, inputs)
            loss = optax.squared_error(predictions, targets).mean()
            acc = jnp.abs(predictions - targets).mean()
            return loss, acc

            # Training function
        def train_step(state, batch):
            def loss_fn(params): return calculate_loss(params, batch)
            ret, grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            loss, acc = ret[0], ret[1]
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss, 'acc': acc}
            return state, metrics

        # Evaluation function
        def eval_step(state, batch):
            loss, acc = calculate_loss(state.params, batch)
            return {'acc': acc, 'loss': loss}

        return train_step, eval_step

    def on_validation_epoch_end(self, epoch_idx, eval_metrics, val_loader):
        if self.trial:
            self.trial.report(eval_metrics['val/acc'], step=epoch_idx)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
