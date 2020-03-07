# tweets-pre-processing
This is a python library performing preprocessing for a sentiment analysis task with a CNN + Embedding model

## Usage
1. Use `from text_pre_processing.text_pre_processing import PreProcessor` to import the `PreProcessor` class

2. The `PreProcessor` takes `max_length_tweet`, and two kwargs: `max_length_tweet`, which has a defautl value of 256, and `max_length_dictionary`, which has a default value of `400002`.
