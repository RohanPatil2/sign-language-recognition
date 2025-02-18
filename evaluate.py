import os
import sys
import glob
import argparse
import json
import logging
import time
import traceback
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import yaml
from generate_keypoints import process_video
from models import Transformer
from configs import TransformerConfig
from utils import load_json, load_label_map
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    data_dir: str
    save_dir: str = "keypoints_dir"
    max_frame_len: int = 169
    frame_length: int = 1080
    frame_width: int = 1920
    batch_size: int = 1
    num_workers: int = 4
    model_checkpoint: str = "path/to/model.pth"
    label_map: str = "include50"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: str = "INFO"
    benchmark: bool = False

class ErrorHandler:
    @staticmethod
    def handle_error(e: Exception, context: str = ""):
        logger.error(f"Error in {context}: {str(e)}")
        logger.debug(traceback.format_exc())
        if isinstance(e, (IOError, FileNotFoundError)):
            logger.error("File operation failed. Check paths and permissions")
        elif isinstance(e, RuntimeError):
            logger.error("Runtime error occurred, possibly CUDA related")

class Benchmark:
    def __init__(self):
        self.timings = {}
        
    def track(self, name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                self.timings[name] = time.time() - start
                logger.info(f"{name} completed in {self.timings[name]:.2f}s")
                return result
            return wrapper
        return decorator

class EnhancedKeypointsDataset(data.Dataset):
    def __init__(self, config: AppConfig):
        try:
            self.files = sorted(glob.glob(os.path.join(config.save_dir, "*.json")))
            if not self.files:
                raise FileNotFoundError(f"No JSON files found in {config.save_dir}")
                
            self.config = config
            self.label_map = load_label_map(config.label_map)
            logger.info(f"Loaded dataset with {len(self.files)} samples")
            
        except Exception as e:
            ErrorHandler.handle_error(e, "dataset initialization")
            raise

    def _validate_keypoints(self, arr: np.ndarray) -> bool:
        """Validate keypoints quality"""
        if np.isnan(arr).mean() > 0.5:
            logger.warning("High NaN ratio in keypoints")
            return False
        if np.var(arr) < 1e-5:
            logger.warning("Low variance in keypoints")
            return False
        return True

    def interpolate(self, arr: np.ndarray) -> np.ndarray:
        try:
            # Advanced interpolation with outlier detection
            df = pd.DataFrame(arr.squeeze())
            df = df.interpolate(method='time', limit_direction='both')
            
            # Handle remaining NaNs
            if df.isna().sum().sum() > 0:
                logger.warning("NaNs detected after interpolation")
                df = df.ffill().bfill()
                
            arr = df.to_numpy()
            return arr.reshape((-1, arr.shape[1]//2, 2))
            
        except Exception as e:
            ErrorHandler.handle_error(e, "interpolation")
            return np.zeros_like(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            file_path = self.files[idx]
            row = pd.read_json(file_path, typ="series")
            
            # Process keypoints with validation
            pose = self._process_landmarks(row.pose_x, row.pose_y)
            h1 = self._process_landmarks(row.hand1_x, row.hand1_y)
            h2 = self._process_landmarks(row.hand2_x, row.hand2_y)
            
            if not all([self._validate_keypoints(x) for x in [pose, h1, h2]]):
                logger.warning(f"Invalid keypoints in {file_path}")
                
            # Create final tensor
            final_data = self._create_tensor(pose, h1, h2)
            return {
                "uid": row.uid,
                "data": torch.FloatTensor(final_data),
                "file_path": file_path
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, f"processing {file_path}")
            return self.__getitem__(idx + 1)  # Skip bad sample

class InferencePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.benchmark = Benchmark()
        self.device = torch.device(config.device)
        self._setup_model()
        
    @Benchmark.track("model_setup")
    def _setup_model(self):
        try:
            self.model = Transformer(
                config=TransformerConfig(size="large", max_position_embeddings=256),
                n_classes=50
            ).to(self.device)
            
            checkpoint = torch.load(self.config.model_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            logger.info(f"Loaded model from {self.config.model_checkpoint}")
            
            if self.config.device == "cuda":
                logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
        except Exception as e:
            ErrorHandler.handle_error(e, "model loading")
            raise

    @Benchmark.track("data_processing")
    def process_videos(self):
        try:
            if os.path.isdir(self.config.save_dir):
                shutil.rmtree(self.config.save_dir)
            os.makedirs(self.config.save_dir, exist_ok=True)
            
            video_paths = glob.glob(os.path.join(self.config.data_dir, "*"))
            logger.info(f"Processing {len(video_paths)} videos")
            
            for path in tqdm(video_paths, desc="Processing Videos"):
                try:
                    process_video(path, self.config.save_dir)
                except Exception as e:
                    logger.error(f"Failed to process {path}")
                    continue
                    
        except Exception as e:
            ErrorHandler.handle_error(e, "video processing")
            raise

    @torch.no_grad()
    @Benchmark.track("inference")
    def run_inference(self, dataloader: data.DataLoader) -> List[Dict]:
        self.model.eval()
        predictions = []
        
        try:
            for batch in tqdm(dataloader, desc="Evaluation"):
                try:
                    input_data = batch["data"].to(self.device)
                    output = self.model(input_data)
                    probs = torch.softmax(output, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
                    
                    predictions.append({
                        "uid": batch["uid"][0],
                        "predicted_label": self.label_map[preds[0]],
                        "confidence": float(probs[0].max()),
                        "timestamp": time.time(),
                        "file_path": batch["file_path"][0]
                    })
                    
                except Exception as e:
                    logger.error(f"Failed inference for {batch.get('uid', 'unknown')}")
                    continue
                    
        except Exception as e:
            ErrorHandler.handle_error(e, "inference loop")
            raise
            
        return predictions

def load_config(config_path: str) -> AppConfig:
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return AppConfig(**config_data)
    except Exception as e:
        ErrorHandler.handle_error(e, "config loading")
        sys.exit(1)

def main():
    try:
        # Parse arguments and load config
        parser = argparse.ArgumentParser(description="Sign Language Recognition Pipeline")
        parser.add_argument("--config", default="config.yaml", help="Path to config file")
        args = parser.parse_args()
        
        config = load_config(args.config)
        logger.setLevel(config.log_level)
        
        # Initialize pipeline
        pipeline = InferencePipeline(config)
        
        # Run processing stages
        pipeline.process_videos()
        dataset = EnhancedKeypointsDataset(config)
        
        dataloader = data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=lambda x: x[0]  # Handle skipped samples
        )
        
        predictions = pipeline.run_inference(dataloader)
        
        # Save results with metadata
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": vars(config),
            "predictions": predictions,
            "benchmark": pipeline.benchmark.timings
        }
        
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        ErrorHandler.handle_error(e, "main pipeline")
        sys.exit(1)

if __name__ == "__main__":
    main()