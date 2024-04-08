
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cross-domain Switch-aware Re-parameterization Visual Geo-Loclization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference.")
    parser.add_argument("--epochs_num", type=int, default=10,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00005, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"], help="_")
    parser.add_argument("--aggregation", type=str, default="salad", choices=["salad"])
    parser.add_argument("--num_trainable_blocks", type=int, default=4,
                        help="num_trainable_blocks")
    parser.add_argument("--features_dim", type=int, default=8448,
                        help="features_dim")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@1.")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default="/mnt/sda3/Projects/npr/datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()

    return args