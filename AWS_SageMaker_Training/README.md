# Sentiment Analysis AWS SageMaker Training
This project enables user to trian a cnn model on AWS SageMaker instance, loading data from AWS S3.

## Changes from Local Training Version
The AWS SageMaker instance training version can only load one file each of the time from S3. Relative path to the S3 bucket needs to be provided for the `--train`, `--validation`, and `--eval` args, as well as the bucket name where the data stored(`--bucket`).

Changes:
1. Added `--bucket` arg to take the S3 bucket name in `sentiment_training.py`.
2. Loaded `boto3` and several other packages in `sentiment_dataset.py` in order to communicate with S3.
3. Changed `_input_fn` function in `sentiment_dataset.py` in order to read data from S3. 

## Usage
1. Open a new terminal in AWS SageMaker Jupyter Notebook;

2. Enter into the root dir of the project;

3. Run `python sentiment_training.py` with arguments:

    `--bucket`: bucket name of the AWS S3 bucket where the data stored.
    
    `--train`: relative path of the training data file on S3 to the bucket;

    `--validation`: relative path of the dev data file on S3 to the bucket;

    `--eval`: relative path of the evaluation data file on S3 to the bucket;

## Example
Data srorage on AWS S3:
```bash
my-bucket
├── training_data
│   └── data-train.json
├── validation_data
│   └── data-validation.json
└── evaluation_data
    └── data-eval.json
```
Use `python sentiment_training.py --bucket my-bucket --train training_data/data-train.json --validation validation_data/data-validation.json --eval evaluation_data/data-eval.json` to train the model.
