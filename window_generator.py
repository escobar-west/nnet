import polars as pl
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        label_columns: list[str],
        norm_flag: bool = False,
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.norm_flag = norm_flag

        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
                f"Norm flag: {self.norm_flag}",
            ]
        )

    def split_window(self, features: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1,
        )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def apply_norm(self, x: tf.Tensor, input_width: int) -> tf.Tensor:
        if self.norm_flag == False:
            return x
        input = x[..., :input_width, :]
        mean = tf.math.reduce_mean(input, axis=-2, keepdims=True)
        stdev = tf.math.reduce_std(input, axis=-2, keepdims=True)
        return (x - mean) / stdev

    def make_dataset(self, data: pl.DataFrame) -> tf.data.Dataset:
        data = data.to_numpy(order="c")
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )
        ds = ds.map(lambda x: self.split_window(self.apply_norm(x, self.input_width)))
        return ds

    def compile_and_fit(self, model: tf.keras.Model, patience=50):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.legacy.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )
        history = model.fit(
            self.train,
            epochs=1000,
            verbose=0,
            validation_data=self.val,
            callbacks=[early_stopping],
        )
        return history

    def plot(
        self,
        model: tf.keras.Model = None,
        plot_col="Real_30d_Vol",
        max_subplots: int = 3,
        use_test=False,
    ):
        dataset = self.test if use_test else self.train
        inputs, labels = next(iter(dataset))
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col}")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )
            label_col_index = self.label_columns_indices.get(plot_col)
            if label_col_index is None:
                continue
            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )
            if n == 0:
                plt.legend()
        plt.xlabel("Time [h]")

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
