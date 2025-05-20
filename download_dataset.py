import os
import json
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

class KaggleDatasetDownloader:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.dataset_dir = Path('dataset')
        self.raw_data_dir = self.dataset_dir / 'raw'
        
    def download_dataset(self):
        """Download the MS2MultiDocuments dataset from Kaggle."""
        print("Downloading dataset...")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.api.dataset_download_files(
            'mathurinache/ms2multidocuments',
            path=str(self.raw_data_dir),
            unzip=True
        )
        print("Dataset downloaded successfully!")
    
    def process_dataset(self):
        """Process the downloaded dataset and split it into train/test/validation."""
        print("Processing dataset...")
        
        # Load the dataset
        df = pd.read_json(self.raw_data_dir / 'ms2multidocuments.json')
        
        # Create train/test/validation splits (80/10/10 split)
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split the data
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Create directories for each split
        train_dir = self.dataset_dir / 'train'
        val_dir = self.dataset_dir / 'validation'
        test_dir = self.dataset_dir / 'test'
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Save each split as separate JSON files
        train_df.to_json(train_dir / 'train_data.json', orient='records', lines=True)
        val_df.to_json(val_dir / 'validation_data.json', orient='records', lines=True)
        test_df.to_json(test_dir / 'test_data.json', orient='records', lines=True)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")

def main():
    """Download and process the dataset."""
    downloader = KaggleDatasetDownloader()
    downloader.download_dataset()
    downloader.process_dataset()

if __name__ == "__main__":
    main()
