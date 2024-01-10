import tensorflow as tf


class Linear(tf.keras.Model):
    def __new__(_cls):
        return tf.keras.Sequential(
            [tf.keras.layers.Dense(units=1, activation="linear")]
        )
