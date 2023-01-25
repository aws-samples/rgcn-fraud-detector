import argparse
import json
import logging
import os
import sys

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

import json

from io import BytesIO
from io import StringIO
from botocore.response import StreamingBody

from fraud_detector import FraudRGCN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    model = FraudRGCN.load_fg(model_dir)
    return model

def predict_fn(input_data, model):
    logger.info(input_data)
    with torch.no_grad():
        fraud_proba=model.predict(input_data, k=2)
        return fraud_proba

def input_fn(
  serialized_input_data: StreamingBody,
  content_type: str = "application/x-parquet",
) -> pd.DataFrame:
    """Deserialize inputs"""
    if content_type == "application/x-parquet":
        data = BytesIO(serialized_input_data)
        df = pd.read_parquet(data)
        return df
    
    elif content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(serialized_input_data), 
                         header=0, sep=",")

        return df

    else:
        raise ValueError("{} not supported by script!".format(content_type))


def train(args):
    
    import warnings
    ### disable CUDA-related warnings from torch library 
    warnings.filterwarnings("ignore", category=UserWarning)    
    
    fd=FraudRGCN()
    
    params = {
        'embedding_size': args.embedding_size,
        'n_layers': args.n_layers,
        'n_epochs': args.n_epochs,
        'n_hidden': args.n_hidden,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'lr': args.lr,
    }
    
    ### read training data
    df_train = pd.read_parquet(args.data_dir)
    
    ### convert df with train transactions to CSV file 
    ### maybe needed to handle columns with all NaNs when envoking endpoint with CSV
    ### df_train=pd.read_csv(StringIO(df_train.to_csv(index=False)), header=0)
    
    ### train model in inductive mode
    fd.train_fg(df_train, params=params)
    
    ### save trained model
    fd.save_fg(args.model_dir)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    # Data and model checkpoints directories
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64,
        help="size of embedding (default: 64)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of layers (default: 2)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=16,
        help="number of hiddent units (default: 16)",
    )
    
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="weight decay (default: 0.2)"
    )
    
    parser.add_argument(
        "--weight_decay", type=float, default=5e-05, help="weight decay (default: 5e-05)"
    )

    
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )

    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    train(parser.parse_args())
