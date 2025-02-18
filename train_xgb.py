import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from models import Xgboost
from configs import XgbConfig
from utils import get_experiment_name, load_label_map
from augment import (
    plus7rotation,
    minus7rotation,
    gaussSample,
    cutout,
    upsample,
    downsample,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def flatten(arr, max_seq_len=200):
    """Pads and flattens an array to a fixed sequence length."""
    arr = np.array(arr)
    arr = np.pad(arr, ((0, max_seq_len - arr.shape[0]), (0, 0)), "constant")
    return arr.flatten()


def combine_xy(x, y):
    """Combines X and Y coordinates into a single feature array."""
    x, y = np.array(x), np.array(y)
    x = x.reshape((-1, x.shape[1], 1))
    y = y.reshape((-1, y.shape[1], 1))
    return np.concatenate((x, y), axis=-1).astype(np.float32)


def split_xy(data):
    """Splits a combined array back into X and Y coordinates."""
    x_values, y_values = zip(*[(row[:, 0], row[:, 1]) for row in data if row.shape != ()])
    return np.array(x_values), np.array(y_values)


def augment_sample(df, augs):
    """Applies multiple augmentations to the data."""
    df = df.copy()
    pose = combine_xy(df.pose_x, df.pose_y)
    hand1 = combine_xy(df.hand1_x, df.hand1_y)
    hand2 = combine_xy(df.hand2_x, df.hand2_y)

    input_df = pd.DataFrame({
        "uid": df.uid,
        "pose": pose.tolist(),
        "hand1": hand1.tolist(),
        "hand2": hand2.tolist(),
        "label": df.label
    })

    augmented_samples = []
    for aug in augs:
        df_aug = aug(input_df)
        pose_x, pose_y = split_xy(df_aug.pose)
        hand1_x, hand1_y = split_xy(df_aug.hand1)
        hand2_x, hand2_y = split_xy(df_aug.hand2)

        augmented_samples.append(pd.Series({
            "uid": f"{df.uid}_{aug.__name__}",
            "label": df.label,
            "pose_x": pose_x.tolist(),
            "pose_y": pose_y.tolist(),
            "hand1_x": hand1_x.tolist(),
            "hand1_y": hand1_y.tolist(),
            "hand2_x": hand2_x.tolist(),
            "hand2_y": hand2_y.tolist(),
            "n_frames": df.n_frames
        }))

    return pd.concat(augmented_samples, axis=0)


def preprocess(df, use_augs, label_map, mode):
    """Processes raw JSON data into feature matrices."""
    feature_cols = ["pose_x", "pose_y", "hand1_x", "hand1_y", "hand2_x", "hand2_y"]
    x, y = [], []

    no_of_videos = df.shape[0] // 9
    pbar = tqdm(total=no_of_videos, desc=f"Processing {mode} dataset")

    for i in range(no_of_videos):
        if use_augs and mode == "train":
            augs = [plus7rotation, minus7rotation, gaussSample, cutout, upsample, downsample]
            augmented_rows = augment_sample(df.iloc[i], augs)
            df = pd.concat([df, augmented_rows], ignore_index=True)

        row = df.loc[i, feature_cols]
        flatten_features = np.hstack(list(map(flatten, row.values)))
        x.append(flatten_features)
        y.append(label_map[df.loc[i, "label"]])
        pbar.update(1)

    pbar.close()
    return np.stack(x), np.array(y)


def load_dataframe(files):
    """Loads and concatenates multiple JSON files into a Pandas DataFrame."""
    if not files:
        raise FileNotFoundError("No data files found!")

    series_list = [pd.read_json(file, typ="series") for file in files]
    return pd.concat(series_list, axis=0, keys=range(len(series_list)))


def fit(args):
    """Training pipeline for the model."""
    train_files = sorted(glob.glob(os.path.join(args.data_dir, f"{args.dataset}_train_keypoints", "*.json")))
    val_files = sorted(glob.glob(os.path.join(args.data_dir, f"{args.dataset}_val_keypoints", "*.json")))

    if not train_files or not val_files:
        raise FileNotFoundError("Training/Validation data files are missing!")

    logging.info("Loading training and validation data...")
    train_df = load_dataframe(train_files)
    val_df = load_dataframe(val_files)

    label_map = load_label_map(args.dataset)
    x_train, y_train = preprocess(train_df, args.use_augs, label_map, "train")
    x_val, y_val = preprocess(val_df, args.use_augs, label_map, "val")

    config = XgbConfig()
    model = Xgboost(config=config)
    
    logging.info("### Training XGBoost Model ###")
    model.fit(x_train, y_train, x_val, y_val)

    exp_name = get_experiment_name(args)
    save_path = os.path.join(args.save_path, f"{exp_name}.pickle.dat")
    model.save(save_path)
    logging.info(f"Model saved at {save_path}")


def evaluate(args):
    """Evaluation pipeline for the trained model."""
    test_files = sorted(glob.glob(os.path.join(args.data_dir, f"{args.dataset}_test_keypoints", "*.json")))

    if not test_files:
        raise FileNotFoundError("Test data files are missing!")

    logging.info("Loading test data...")
    test_df = load_dataframe(test_files)

    label_map = load_label_map(args.dataset)
    x_test, y_test = preprocess(test_df, args.use_augs, label_map, "test")

    exp_name = get_experiment_name(args)
    config = XgbConfig()
    model = Xgboost(config=config)

    load_path = os.path.join(args.save_path, f"{exp_name}.pickle.dat")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found at {load_path}")

    logging.info("### Loading Trained Model ###")
    model.load(load_path)

    logging.info("### Running Model Evaluation ###")
    test_preds = model.predict(x_test)
    acc = accuracy_score(y_test, test_preds)

    logging.info(f"Test Accuracy: {acc:.4f}")
    return acc
