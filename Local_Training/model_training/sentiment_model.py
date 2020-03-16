"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def select_model(name, embedding_matrix, config):
    candidate_models = {
        "CNN": keras_model_cnn,
        "RNN": keras_model_rnn,
        "CnnLstm": keras_model_CnnLstm
    }
    return candidate_models[name](embedding_matrix, config)

def keras_model_cnn(embedding_matrix, config):
    """
    Creating a CNN model for sentiment modeling

    """
    cnn_model = models.Sequential()
    cnn_model.add(
        layers.Embedding(
            input_length = config['padding_size'],
            input_dim = config['embeddings_dictionary_size'],
            output_dim = config['embeddings_vector_size'],
            weights = [embedding_matrix],
            trainable = True,
            name = 'embedding')
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
    cnn_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])
    return cnn_model

def keras_model_rnn(embedding_matrix, config):
    """
    Creating a RNN model for sentiment modeling

    """
    rnn_model = models.Sequential()
    rnn_model.add(
        layers.Embedding(
            input_length = config['padding_size'],
            input_dim = config['embeddings_dictionary_size'],
            output_dim = config['embeddings_vector_size'],
            weights = [embedding_matrix],
            trainable = True,
            name = 'embedding')
    )
    rnn_model.add(
        layers.SimpleRNN(
            100
        )
    )
    rnn_model.add(
        layers.Dense(
            units = 10,
            activation='relu')
    )
    rnn_model.add(
        layers.Dense(
            units = 1,
            activation='sigmoid')
    )
    rnn_model.compile(
        optimizer = 'adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])
    return rnn_model

def keras_model_CnnLstm(embedding_matrix, config):
    """
    Creating a LSTM model for sentiment modeling

    """
    model = models.Sequential()
    model.add(
        layers.Embedding(
            input_length = config['padding_size'],
            input_dim = config['embeddings_dictionary_size'],
            output_dim = config['embeddings_vector_size'],
            weights = [embedding_matrix],
            trainable = True,
            name = 'embedding')
    )
    model.add(
        layers.Dropout(0.25)
    )
    model.add(
        layers.Conv1D(
            64,
            5,
            padding='valid',
            activation='relu',
            strides=1)
    )
    model.add(
        layers.MaxPooling1D(
            pool_size=4)
    )
    model.add(
        layers.LSTM(64)
    )
    model.add(
        layers.Dense(1)
    )
    model.add(
        layers.Activation('sigmoid')
    )
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
    
def fit_model(
    model_name, 
    embedding_matrix, 
    config, 
    train_dataset,
    validation_dataset,
    eval_dataset
    ):
    model = select_model(model_name, embedding_matrix, config)

    print(f"Starting training {model_name}...")

    model.fit(
        x=train_dataset[0],
        y=train_dataset[1],
        steps_per_epoch=train_dataset[2]["num_batches"],
        epochs=config["num_epoch"],
        validation_data=(validation_dataset[0], validation_dataset[1]),
        validation_steps=validation_dataset[2]["num_batches"]
    )

    score = model.evaluate(
        eval_dataset[0],
        eval_dataset[1],
        steps=eval_dataset[2]["num_batches"],
        verbose=0
    )
        
    print(f"Finished trainig {model_name} model:")
    print("Test loss:{}".format(score[0]))
    print("Test accuracy:{}".format(score[1]))

    return model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """
    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))
