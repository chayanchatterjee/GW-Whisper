import argparse
from src.sd_train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline for LIGO binary classification model',
                                     epilog='This script trains a binary classification model with whisper encoder, make sure the preprocess.py script has been run before running this script. \
                                        Example of usage: python train.py --data-path /path/to/data --results-path /path/to/results --log-dir /path/to/logs --batch-size 32 --num-epochs 100 --learning-rate 3e-5 --seed 42 --num-workers 4 --method DoRA --lora-rank 8 --lora-alpha 32')
    
    parser.add_argument('--data-path', type=str, default='./data/Whisper_train_mass-8to100_resampled_train', help='Path to the dataset')
    parser.add_argument('--models-path', type=str, default='./results/Single_detector/models', help='Path to save the model weights')
    parser.add_argument('--figures-path', type=str, default='./results/Single_detector/figures', help='Path to save the figures')
    parser.add_argument('--log-dir', type=str, default='./results/Single_detector/logs', help='Directory to save the TensorBoard logs')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--load_lora_weights', type=str, default=None)
    parser.add_argument('--load_dense_weights', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size')
    parser.add_argument('--encoder', type=str, default='tiny', help='Encoder to use (small, base, or large-v1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--method', type=str, default='DoRA', help='Select Which model to train, options are: full_finetune, LoRA, DoRA')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
  
    args = parser.parse_args()
    
    main(args)
