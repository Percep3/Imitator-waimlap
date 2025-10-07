import h5py
import json
import os
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
    def __init__(self, h5Path, n_keypoints=111, transform=None, return_label=False, max_length=4000, data_augmentation=True, labels_vocab_path=None):
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

        self.labels_vocab_path = labels_vocab_path or (os.path.splitext(h5Path)[0] + "_labels_vocab.json")
        self.label_to_id = {}
        self.id_to_label = []

        if self.return_label:
            self._build_or_load_label_vocab()

    def _build_or_load_label_vocab(self):
        # Si ya existe, cargar
        if os.path.exists(self.labels_vocab_path):
            with open(self.labels_vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.id_to_label = data["id_to_label"]
            self.label_to_id = {s: i for i, s in enumerate(self.id_to_label)}
            return

        # Si no existe, recorrer solo los índices válidos y recolectar labels
        labels_set = set()
        with h5py.File(self.h5Path, 'r') as f:
            for (dataset, clip) in self.valid_index:
                s = f[dataset]["labels"][clip][:][0].decode()
                labels_set.add(s)

        # Vocab ordenado para estabilidad
        self.id_to_label = sorted(labels_set)
        self.label_to_id = {s: i for i, s in enumerate(self.id_to_label)}

        # Guardar a disco (recomendado)
        with open(self.labels_vocab_path, "w", encoding="utf-8") as f:
            json.dump({"id_to_label": self.id_to_label}, f, ensure_ascii=False, indent=2)

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
                                  return_label=self.return_label,
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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
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
        label_id = None

        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
    
            if self.return_label:
                label_str = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()
                label_id = self.label_to_id[label_str]

        #keypoint = remove_keypoints(keypoint)
        keypoint = normalize_augment_data(keypoint, "Original", self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        if self.return_label:
            # print("retornando label ", label)
            return keypoint, embedding, label_id

        return keypoint, embedding, None