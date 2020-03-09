"""
Main sentiment training script

Author pharnoux


"""

import os
import argparse
import sentiment_dataset as sentiment_dataset
import sentiment_model as sentiment_model
import config_holder as config_holder

def main(args):
    """
    Main training method

    """

    print("Preparing for training...")

    training_config = config_holder.ConfigHolder(args.config_file).config

    training_config["num_epoch"] = args.num_epoch
    
    embedding_matrix = sentiment_dataset._load_embedding_matrix(training_config)

    models = {}
    for model_name in training_config["models"]:
        train_dataset = sentiment_dataset.train_input_fn(args.train, training_config)
        validation_dataset = sentiment_dataset.validation_input_fn(args.validation, training_config)
        eval_dataset = sentiment_dataset.eval_input_fn(args.eval, training_config)

        model = sentiment_model.fit_model(
            model_name, 
            embedding_matrix, 
            training_config, 
            train_dataset,
            validation_dataset,
            eval_dataset
        )
        models[model_name] = model
        
        sentiment_model.save_model(model, os.path.join(args.model_output_dir, f"sentiment_model_{model_name}_non_stopwords_5.h5"))
        print("{} model saved to {}".format(model_name, os.path.join(args.model_output_dir, f"sentiment_model_{model_name}.h5")))
def get_arg_parser():
    """
    Adding this method to unit test

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="../data/training_non_stopwords",
        help="The directory where the training data is stored.")
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default="../data/validation_non_stopwords",
        help="The directory where the validation data is stored.")
    parser.add_argument(
        "--eval",
        type=str,
        required=False,
        default="../data/evaluation_non_stopwords",
        help="The directory where the evalutaion data is stored.")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=False,
        default="models")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
        help="The number of steps to use for training.")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config.json"),
        help="The path to the training config file.")

    return parser

if __name__ == "__main__":
    PARSER = get_arg_parser()
    ARGS = PARSER.parse_args()
    main(ARGS)
