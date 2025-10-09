from settings import initialize
initialize()

import torch
from src.mslm.utils.setup_train import setup_paths, prepare_datasets
from src.mslm.training.cross_val import cross_validate
from src.mslm.utils.config_loader import cfg

def run(
    epochs: int,
    batch_size: int,
    batch_sample: int,
    checkpoint_interval: int,
    log_interval: int,
    train_ratio: float = 0.8,
    key_points: int = 111,
    batch_sampling: bool = True,
    n_folds=5
    ):
    _, _, h5_file = setup_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- config de entrenamiento ---
    training_cfg:dict = cfg.training
    model_cfg = cfg.model
    
    train_ratio = training_cfg.get("train_ratio", train_ratio)
    training_cfg.update({
        "epochs": epochs if epochs else training_cfg.get("epochs", 100),
        "batch_size": batch_size if batch_size else training_cfg.get("batch_size", 32),
        "batch_sample": batch_sample if batch_sample else training_cfg.get("sub_batch_size", 32),
        "checkpoint_interval": checkpoint_interval if checkpoint_interval else training_cfg.get("checkpoint_interval", 10000),
        "log_interval": log_interval if log_interval else training_cfg.get("log_interval", 2),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_cfg.get("device") == "auto" else model_cfg.get("device", device),
        "n_keypoints": key_points,
    })
    print(training_cfg)

    if batch_sampling:
        if batch_size%batch_sample != 0 or batch_size < batch_sample:
            raise ValueError(f"The sub_batch {batch_sample} needs to be divisible the batch size {batch_size}")

    training_cfg["batch_sampling"] = batch_sampling
    training_cfg["batch_sample"] = batch_sample
    training_cfg["compile"] = True

    print(f"Batch size: {batch_size}, batch sample: {batch_sample}")
    print(f"using dataset {h5_file}")

    # Prepare dataset
    dataset, _, _, _ = prepare_datasets(h5_file, train_ratio=1.0)
    
    # Perform cross validation
    results = cross_validate(
        dataset=dataset,
        n_splits=n_folds,
        model_params=model_cfg,
        training_params=training_cfg
    )
    
    # Print results
    print("\nCross Validation Results:")
    print(f"Average Train Loss: {results['avg_train_loss']:.4f}")
    print(f"Average Val Loss: {results['avg_val_loss']:.4f}")
    
    for i, metrics in enumerate(results['fold_metrics']):
        print(f"\nFold {i+1}:")
        print(f"Train Loss: {metrics['train_loss']:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--batch_sample", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging training progress.")
    parser.add_argument("--num_keypoints", type=int, default=111, help="Number of keypoints to use in the model.")
    parser.add_argument("--batch_sampling", type=bool, default=False, help="Enables batch sampling for training.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross validation")
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.batch_sample, args.log_interval, args.batch_sampling, args.num_keypoints, args.n_folds)