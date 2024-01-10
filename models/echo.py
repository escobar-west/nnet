import tensorflow as tf


class EchoModel(tf.keras.Model):
    """
    Uses current value as prediction for future values
    """

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs: tf.Tensor):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
