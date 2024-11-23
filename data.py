from torch.utils.data import Dataset, DataLoader
import polars as pl
import os
import torch
from pathlib import Path
import numpy as np
import typing as tp


def _get_df(path: str | Path):
    if not os.path.exists(path) and not os.path.isfile(path) and not path.endswith('parquet'):
        raise Exception('Incorrect path provided')

    return pl.read_parquet(path)\
        .with_row_index()


def collate_fn(batch):
    if isinstance(batch, pl.dataframe.frame.DataFrame):
        return pl.vstack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)


class PCDataset(Dataset):
    def __init__(
        self,
        data_file: Path | str,
        input_size=48,
        out_size=24,
        cols=['Value', 'temp', 'feels_like', 'pressure', 'wind_speed'],
        transforms: list[tp.Callable] = []
    ) -> None:
        super().__init__()
        self.df = _get_df(data_file)
        self.out_size = out_size
        self.input_size = input_size
        self.coi = cols
        self.df = self.df.select(self.coi + ['index']).drop_nulls()
        self.input_data = self.df.drop('index').to_jax()
        self.targets = self.df.select(pl.col('Value')).to_jax()
        self.transforms = transforms

    def __getitem__(self, index):
        '''
            index - position of input
        '''

        npt = self.input_data[index:(index+self.input_size)]
        return npt, self.targets[(index+self.input_size):(index+self.input_size+self.out_size)]

    def __len__(self):
        return len(self.df) - (self.out_size + self.input_size) + 1


class PCDataModule():
    def __init__(self,
                 data_dir: Path,
                 input_size: int = 48,
                 out_size: int = 24,
                 batch_size: int = 4,
                 coi: list[str] = ['Value', 'temp',
                                   'feels_like', 'pressure', 'wind_speed'],
                 transforms: list[tp.Callable] = []) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.out_size = out_size
        self.coi = coi
        self.transforms = transforms
        self.setup('all')

    def setup(self, stage: str) -> None:
        self.train_ds = PCDataset(self.data_dir / 'train_pc.parquet',
                                  input_size=self.input_size,
                                  out_size=self.out_size,
                                  cols=self.coi)
        self.test_ds = PCDataset(self.data_dir / 'test_pc.parquet',
                                 input_size=self.input_size,
                                 out_size=self.out_size,
                                 cols=self.coi)
        self.val_ds = PCDataset(self.data_dir / 'val_pc.parquet',
                                input_size=self.input_size,
                                out_size=self.out_size,
                                cols=self.coi)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
