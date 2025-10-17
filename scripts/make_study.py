from settings import initialize
initialize()

import optuna
import torch
import random

torch.manual_seed(23)
random.seed(23)

from src.mslm.utils.setup_train import setup_paths, prepare_datasets, create_dataloaders
from src.mslm.studies import complete_objective
from src.mslm.utils.config_loader import cfg

def run(
    epochs: int = 15,
    n_trials: int = 10,
    batch_size: int = 32,
    batch_sample: int = 8,
    train_ratio: float = 0.8,
    key_points: int = 111,
    batch_sampling: bool = True
):
    # setup
    _, _, h5_file = setup_paths()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- config de entrenamiento ---
    training_cfg:dict = cfg.training
    model_cfg = cfg.model

    # --- config de entrenamiento ---
    train_ratio = training_cfg.get("train_ratio", train_ratio)
    training_cfg.update({
        "epochs": epochs if epochs else training_cfg.get("epochs", 100),
        "batch_size": batch_size if batch_size else training_cfg.get("batch_size", 32),
        "batch_sample": batch_sample if batch_sample else training_cfg.get("sub_batch_size", 32),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_cfg.get("device") == "auto" else model_cfg.get("device", device),
    })

    training_cfg["batch_sampling"] = batch_sampling
    training_cfg["batch_sample"] = batch_sample
    training_cfg["compile"] = False

    if batch_sampling:
        if batch_size%batch_sample != 0 or batch_size < batch_sample:
            raise ValueError(f"The sub_batch {batch_sample} needs to be divisible the batch size {batch_size}")

    print(f"Running study with batch size {batch_size}, sub batch size {batch_sample}")

    # datasets
    print(f"Batch size: {batch_size}, batch sample: {batch_sample}")
    print(f"using dataset {h5_file}")
    tr_ds, val_ds, _, _ = prepare_datasets(h5_file, train_ratio, key_points)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=10)

    # optuna
    storage = f"sqlite:///study_models.db"
    study = optuna.create_study(study_name=f"model_{training_cfg['model_version']}_{training_cfg['checkpoint']}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(
        lambda t: complete_objective(t,
            model_params=model_cfg,
            train_dataloader=tr_dl,
            val_dataloader=val_dl,
            train_config=training_cfg,
        ),
        n_trials=n_trials
    )
    print(study.best_trial)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Study a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--n_trials", type=int, default=8, help="Number of trials to run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--batch_sample", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--num_keypoints", type=int, default=111, help="Number of keypoints to use in the model.")
    parser.add_argument("--batch_sampling", type=bool, default=True, help="Enables batch sampling for training.")
    args = parser.parse_args()

    print(f"Running study with {args.n_trials} trials, batch size {args.batch_size}, train ratio {args.train_ratio}")
    run(args.epochs, args.n_trials, args.batch_size, args.batch_sample, args.train_ratio, args.num_keypoints, args.batch_sampling)