# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# A copy of the custom L1 Dist layer used to create the model.
class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
