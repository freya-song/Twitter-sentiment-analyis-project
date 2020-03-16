"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """
    cnn_model = models.Sequential()
    cnn_model.add(
        layers.Embedding(
            input_length = config['padding_size'],
            input_dim = config['embeddings_dictionary_size'],
            output_dim = config['embeddings_vector_size'],
            trainable = True)
    )
    cnn_model.add(
        layers.Conv1D(
            filters = 100,
            kernel_size = 2,
            strides = 1,
            padding = 'valid',
            activation='relu')
    )
    cnn_model.add(
        layers.GlobalMaxPool1D()
    )
    cnn_model.add(
        layers.Dense(
            units = 100,
            activation='relu')
    )
    cnn_model.add(
        layers.Dense(
            units = 1,
            activation='sigmoid')
    )
    cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))
