import h5py
import numpy as np
import torch
from typing import Optional, List, Tuple
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset
from .data_augmentation import normalize_augment_data, remove_keypoints
class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform_fn: str, return_label=False, video_lengths=[], n_keypoints=133):
        self.subset    = subset
        self.transform = transform_fn
        self.return_label = return_label
        self.video_lengths = video_lengths
        self.n_keypoints = n_keypoints
        
        if self.transform == "Length_variance":
            self.video_lengths = [int(round(0.8 * video)) for video in self.video_lengths]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        keypoint, embedding, label = self.subset[idx]

        keypoint = normalize_augment_data(keypoint, self.transform, self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None

class KeypointDataset(Dataset):
    def __init__(self, h5Path, n_keypoints=111, transform=None, return_label=False, max_length=4000, data_augmentation=True):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.transform = transform
        self.return_label = return_label
        self.max_length = max_length
        self.video_lengths = []
        self.data_augmentation = data_augmentation
    
        self.data_augmentation_dict = {
            0: "Length_variance",
            1: "Gaussian_jitter",
            2: "Rotation_2D",
            4: "Scaling"
        }

        self.dataset_length = 0
        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())
            datasets = sorted(datasets)
        
            self.valid_index = []
            self.original_videos = []

            for dataset in datasets:
                #if dataset not in ["dataset1", "dataset3", "dataset5", "dataset7"]:
                #    continue

                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    try:
                        shape = f[dataset]["keypoints"][clip].shape[0]
            
                        if shape < self.max_length:
                            self.valid_index.append((dataset, clip))
                            self.video_lengths.append(shape)
                    except KeyError:
                        print(f"KeyError for {dataset}/{clip}, skipping...")
                        continue
                
            self.dataset_length = len(self.valid_index)

    def split_dataset(self, train_ratio):
        train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 
        
        if self.data_augmentation:
            train_length = [self.video_lengths[i] 
                            for i in train_dataset.indices]
            train_subset = Subset(self, train_dataset.indices)
            aug_subsets = [
                TransformedSubset(train_subset, 
                                  transform_fn=tf,
                                  return_label=False,
                                  video_lengths=train_length,
                                  n_keypoints=self.n_keypoints
                                  )
                for tf in self.data_augmentation_dict.values()
            ]

            trains_subset_length = [ length
                for subset in aug_subsets
                for length in subset.video_lengths
            ]
            
            train_lengths = train_length + trains_subset_length 
            train_dataset = ConcatDataset([train_subset, *aug_subsets])
            
            self.dataset_length = len(val_length) + len(train_length)
        else:
            train_lengths = [self.video_lengths[i] for i in train_dataset.indices]

        print("Videos: ", self.dataset_length)
        return train_dataset, validation_dataset, train_lengths, val_length

    def get_video_lengths(self):
        return self.dataset_length 
    
    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[np.ndarray], torch.Tensor, Optional[str]]:
        """
        Recupera una muestra individual del conjunto de datos.
        Este método recupera los puntos clave, la matriz de adyacencia, los embeddings y opcionalmente las etiquetas
        del archivo HDF5 para el índice dado. Los puntos clave se procesan eliminando
        puntos específicos y normalizando/aumentando los datos.
        Args:
            idx (int): Índice de la muestra a recuperar.
        Returns:
            Tupla que contiene:
            - keypoint (torch.Tensor): Datos de puntos clave procesados.
            - A (Optional[np.ndarray]): Matriz de adyacencia que representa el grafo esquelético.
            - embedding (torch.Tensor): Vector de embedding para la muestra.
            - label (Optional[str]): Cadena de etiqueta si return_label es True, None en caso contrario.
        """
        
        mapped_idx = self.valid_index[idx]
            
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
    
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        #keypoint = remove_keypoints(keypoint)
        keypoint = normalize_augment_data(keypoint, "Original", self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None