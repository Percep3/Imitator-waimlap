import torch
from torch.utils.data import Subset
from ..utils.setup_train import create_dataloaders, build_model
from .trainer import Trainer

def cross_validate(dataset, n_splits, model_params, training_params, seed=42):
    """
    k-fold cross validation over a torch Dataset.
    - dataset: torch.utils.data.Dataset
    - n_splits: int folds
    - model_params: dict passed to build_model
    - training_params: dict containing at least "batch_size" and "learning_rate"
    """
    torch.manual_seed(seed)

    dataset_size = len(dataset)
    if n_splits < 2 or n_splits > dataset_size:
        raise ValueError("n_splits must be >=2 and <= dataset size")

    # reproducible permutation of indices
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()
    fold_size = dataset_size // n_splits
    fold_metrics = []

    for fold in range(n_splits):
        print(f"\nFold {fold+1}/{n_splits}")
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_splits - 1 else dataset_size

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_dataloader, val_dataloader = create_dataloaders(
            train_subset,
            val_subset,
            batch_size=training_params["batch_size"]
        )

        model = build_model(**model_params)

        trainer = Trainer(
            model,
            train_dataloader,
            val_dataloader,
            save_tb_model=False,
            **training_params
        )

        # trainer.train() must return (train_loss, val_loss) — ensure trainer implementation matches
        train_loss, val_loss = trainer.train()

        fold_metrics.append({'train_loss': train_loss, 'val_loss': val_loss})

    avg_train_loss = sum(m['train_loss'] for m in fold_metrics) / n_splits
    avg_val_loss = sum(m['val_loss'] for m in fold_metrics) / n_splits

    return {'fold_metrics': fold_metrics, 'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss}