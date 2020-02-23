"""
Module holing dataset methods

Author pharnoux

"""

import os
import json
import math
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset

# Additional imports for loading data from S3 and accessing S3
import boto3
from sagemaker import get_execution_role
role = get_execution_role()
s3 = boto3.resource('s3')

def train_input_fn(bucket, training_dir, config):
    return _input_fn(training_dir, config, "train")

def validation_input_fn(bucket, training_dir, config):
    return _input_fn(training_dir, config, "validation")

def eval_input_fn(bucket, training_dir, config):
    return _input_fn(training_dir, config, "eval")

def serving_input_fn(_, config):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _load_json_file(content_object, config):

    features = []
    labels = []
    
    #load file as string, seperated by \n
    file_content = content_object.get()['Body'].read().decode('utf-8')
    
    file = file_content.strip().split('\n')

    for line in file:
        entry = json.loads(line)

        if len(entry["feature"]) != config["padding_size"]:
            raise ValueError(
                "The size of the features of the entry with twitterid {} was not expected".format(
                    entry["twitterid"]))

        labels.append(list(map(lambda x: int(x) / 4, entry["label"])))
        features.append(entry["feature"])

    return features, labels

def _input_fn(bucket, directory, config, mode):

    print("Fetching {} data...".format(mode))

#     all_files = os.listdir(directory)

    all_features = []
    all_labels = []

#     for file in all_files:
#         features, labels = _load_json_file(os.path.join(directory, file), config)
#         all_features += features
#         all_labels += labels
    
    #connect to my S3 bucket
    content_object = s3.Object(bucket, directory)
    
    all_features, all_labels = _load_json_file(content_object, config)

    num_data_points = len(all_features)
    num_batches = math.ceil(len(all_features) / config["batch_size"])

    dataset = Dataset.from_tensor_slices((all_features, all_labels))

    if mode == "train":

        dataset = Dataset.from_tensor_slices((all_features, all_labels))
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
