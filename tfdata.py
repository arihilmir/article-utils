import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import tensorflow_datasets as tfds


class WindowGenerator():
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        files_path,
        train_file_name = 'train_data.parquet',
        test_file_name = 'test_data.parquet',
        cols=['Value', 'feels_like', 'pressure', 'wind_speed'],
        label_col = 'Value',
        batch_size=8,
        train_part = 0.5,
        partial_dataset=False
    ):
        self.extract_labels = False
        # Store the raw data.
        if label_col not in cols:
            cols.append(label_col)
            self.extract_labels = True
        self.training_data = pd.read_parquet(files_path / train_file_name)[cols]
        split_pos = int(len(self.training_data) * train_part)

        if partial_dataset:
            self.training_data = self.training_data.iloc[:350]
            split_pos = int(len(self.training_data) * train_part)

        self.train_df = self.training_data.iloc[:split_pos]
        self.val_df = self.training_data.iloc[split_pos:]

        self.test_df = pd.read_parquet(files_path / test_file_name)[cols]
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = [label_col]
        if cols is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(cols)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        last_elment = -1 if self.extract_labels else len(self.train_df.columns)
        inputs = features[:, self.input_slice, :last_elment]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data: pd.DataFrame):
        data = np.array(data, dtype=np.float32)
        ds = keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
