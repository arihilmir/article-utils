from tqdm.auto import tqdm
# import numpy as np
import jax
import jax.numpy as jnp


class Logger:
    def __init__(self):
        self.values = {}

    def log(self, key: str, value: float):
        step_key = key + '_step'
        epoch_key = key + '_epoch'
        self.values[step_key] = value
        if epoch_key in self.values:
            self.values[epoch_key].append(value)

    def add_loss(self, loss):
        self.epoch_accuracies.append(loss)

    def summarize(self) -> str:
        return ' '.join(f'[{key}: {value:4.6f}]' if value is not list else f'[{key}: {jax.device_get(jnp.array(value).mean()):4.6f}]'
                        for key, value in self.values.items())

    def tqdm_summarize(self) -> dict:
        return {key: f'{value:4.6f}' if value is not list else f'{jax.device_get(jnp.array(value).mean()):4.6f}'
                for key, value in self.values.items()}

    def epoch_end(self):
        self.values = {}
