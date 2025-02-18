import os
import time
import warnings
import argparse
import logging
import torch

import train_nn
import train_xgb
from cnn_runner import save_cnn_features

# Setup logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def validate_args(args):
    """ Validate input arguments to prevent incorrect configurations. """
    if args.model not in ["lstm", "transformer", "xgboost"]:
        raise ValueError(f"Invalid model type: {args.model}. Choose from lstm, transformer, xgboost.")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")

    if args.epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    if args.use_pretrained not in [None, "evaluate", "resume_training"]:
        raise ValueError("Invalid use_pretrained value. Options: None, evaluate, resume_training.")

def get_device():
    """ Detect available device: GPU or CPU. """
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="INCLUDE trainer for XGBoost, LSTM, and Transformer")
    parser.add_argument("--seed", default=0, type=int, help="Seed value for reproducibility")
    parser.add_argument("--dataset", default="include", type=str, choices=["include", "include50"], help="Dataset choice")
    parser.add_argument("--use_augs", action="store_true", help="Use data augmentation")
    parser.add_argument("--use_cnn", action="store_true", help="Use CNN to generate embeddings")
    parser.add_argument("--model", default="lstm", type=str, choices=["lstm", "transformer", "xgboost"], help="Model type")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to train, val, and test JSON files")
    parser.add_argument("--save_path", default="./", type=str, help="Directory to save the trained model")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for neural networks")
    parser.add_argument("--transformer_size", default="small", type=str, choices=["small", "large"], help="Transformer size")
    parser.add_argument("--use_pretrained", default=None, choices=[None, "evaluate", "resume_training"], help="Use pretrained model")
    
    args = parser.parse_args()

    # Validate input arguments
    validate_args(args)

    # Detect device
    device = get_device()
    logging.info(f"Using device: {device}")

    # Measure execution time
    start_time = time.time()

    if args.model == "xgboost":
        if args.use_pretrained:
            raise Exception("Pre-trained models are not available for XGBoost")
        
        if args.use_cnn:
            warnings.warn("use_cnn flag is set for XGBoost, but it does not use CNN features.")
        
        logging.info("### Starting XGBoost Training ###")
        train_xgb.fit(args)
        logging.info("### Evaluating XGBoost Model ###")
        train_xgb.evaluate(args)

    else:
        if args.use_cnn:
            logging.info("Extracting CNN features...")
            try:
                save_cnn_features(args)
                if args.use_augs:
                    warnings.warn("Cannot perform augmentation on CNN features")
            except Exception as e:
                logging.error(f"Error during CNN feature extraction: {e}")
                return
        
        if args.use_pretrained == "evaluate":
            logging.info("### Evaluating Pretrained Model ###")
            train_nn.evaluate(args)
        else:
            logging.info("### Starting Neural Network Training ###")
            train_nn.fit(args)
            logging.info("### Evaluating Trained Model ###")
            train_nn.evaluate(args)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
