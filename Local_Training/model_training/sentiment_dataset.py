"""
Module holing dataset methods

Author pharnoux

"""

import os
import json
import math
import tensorflow as tf
import numpy as np

def train_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "train")

def validation_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "validation")

def eval_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "eval")

def serving_input_fn(_, config):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _load_json_file(json_path, config):

    features = []
    labels = []

    with open(json_path, "r") as file:

        for line in file:

            entry = json.loads(line)

            labels.append(int(entry["sentiment"]) / 4)
            features.append(list(map(lambda x: int(x), entry["feature"])))

    return features, labels

def _input_fn(directory, config, mode):

    print("Fetching {} data...".format(mode))

    all_files = os.listdir(directory)

    all_features = []
    all_labels = []

    for file in all_files:
        features, labels = _load_json_file(os.path.join(directory, file), config)
        all_features += features
        all_labels += labels

    num_data_points = len(all_features)
    num_batches = math.ceil(len(all_features) / config["batch_size"])

    dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))

    if mode == "train":

        dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))
        dataset = dataset.batch(config["batch_size"]).shuffle(10000, seed=12345).repeat(
            config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    if mode in ("validation", "eval"):

        dataset = dataset.batch(config["batch_size"]).repeat(config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    iterator = dataset.make_one_shot_iterator()
    dataset_features, dataset_labels = iterator.get_next()

    return [{config["input_tensor_name"]: dataset_features}, dataset_labels,
            {"num_data_point": num_data_points, "num_batches": num_batches}]

def _load_embedding_matrix(config):
    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))
    print(f"Fetching embedding vectors from {config['embeddings_path']}")
    idx = 0
    with open(config["embeddings_path"], "r", encoding = "utf-8") as file:
        for line in file:
            vector = list(map(float, line.strip().split()[1:]))
            if vector is not None:
                embedding_matrix[idx] = vector
                idx += 1
    return embedding_matrix
