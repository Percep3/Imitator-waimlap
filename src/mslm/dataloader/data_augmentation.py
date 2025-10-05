import numpy as np
import random
import torch

def scaling(keypoint):
    # Escalar los keypoints sin cambiar su cantidad
    scale = random.uniform(0.9, 1.1)
    return keypoint * scale

def horizontal_flip(keypoint):
    # Reflejar horizontalmente la secuencia de keypoints
    return keypoint.flip(dims=[1])  # Reflejo horizontal

def rotation_2D(keypoint):
    # Aseguramos que la rotación no cambie la cantidad de keypoints
    angle = random.uniform(-15, 15)  # Rotación aleatoria en grados
    angle_rad = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)  # Convertir a radianes
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)], 
                                    [torch.sin(angle_rad), torch.cos(angle_rad)]], dtype=torch.float32)
    keypoint_rotated = torch.matmul(keypoint.view(-1, 2), rotation_matrix)
    return keypoint_rotated.view(keypoint.shape)

def length_variance(keypoint, scale=0.8):
    T, J, C = keypoint.shape
    T_new = int(round(T * scale))
    
    orig_times = np.linspace(0, T-1, num=T)
    new_times = np.linspace(0, T-1, num=T_new)

    flat = keypoint.reshape(T, J*C)
    streched = np.stack([
        np.interp(new_times, orig_times, flat[:, d])
        for d in range(J*C)
    ], axis = 1)
    keypoints_streched = streched.reshape(T_new, J, C)
    return keypoints_streched

def gaussian_jitter(keypoint, sigma=0.0014, clip=3.0):
    keypoint_jitter = np.random.normal(loc=0.0, scale=sigma, size=keypoint.shape)

    if clip is not None:
        np.clip(keypoint_jitter, -clip, clip, out=keypoint_jitter)    
    
    return keypoint + keypoint_jitter

def apply_augmentation(keypoint, augmentation_type):
    """
    Aplica diferentes augmentaciones de acuerdo al tipo.
    """
    
    if augmentation_type == "Gaussian_jitter":
        return gaussian_jitter(keypoint)
    elif augmentation_type == "Length_variance":
        return length_variance(keypoint)
    elif augmentation_type == "Rotation_2D":
        return rotation_2D(keypoint)
    elif augmentation_type == "Horizontal_flip":
        return horizontal_flip(keypoint) #TDB
    elif augmentation_type == "Scaling":
        return scaling(keypoint)
    return keypoint

def keypoint_normalization(keypoint):
    keypoint = abs(keypoint)

    mask_keypoints = ~((keypoint[...,0] < 5) & (keypoint[...,1] < 5))
    
    valid_points = keypoint[mask_keypoints].view(-1, 2)
    if valid_points.numel() > 0:
        global_mins, _ = valid_points.min(dim=0)
        global_maxs, _ = valid_points.max(dim=0)
    else:
        global_mins, _ = torch.zeros(2, device=keypoint.device)
        global_maxs, _ = torch.ones(2, device=keypoint.device)
                    
    global_ranges = global_maxs - global_mins
    global_ranges[global_ranges == 0] = 1.0

    normalized = (keypoint - global_mins) / global_ranges

    normalized[~mask_keypoints] = keypoint[~mask_keypoints] 

    return normalized

def filter_unstable_keypoints_to_num(keypoints, keep_n):
    """
    Conserva los 'keep_n' keypoints más estables (con menor varianza temporal).
    """
    
    if keep_n > keypoints.size(1):
        raise ValueError(f"keep_n ({keep_n}) mayor a cantidad de keypoints ({keypoints.size(1)})")

    T, N, _ = keypoints.shape

    # Calcular varianza temporal por keypoint
    var = keypoints.var(dim=0).mean(dim=1)  # (N,)

    # Obtener los índices de los keypoints más estables
    _, indices = torch.topk(-var, k=keep_n)  # usamos -var para orden ascendente
    stable_mask = torch.zeros(N, dtype=torch.bool)
    stable_mask[indices] = True

    # Aplicar la máscara
    filtered = keypoints.clone()
    for i in range(N):
        if not stable_mask[i]:
            filtered[:, i] = 0

    return filtered, stable_mask

def normalize_augment_data(keypoint, augmentation_type, n_keypoints = 133):
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint)    

    # Clean noise 
    keypoint, _ = filter_unstable_keypoints_to_num(keypoint, n_keypoints)

    if augmentation_type != "Original":
        keypoint = apply_augmentation(keypoint, augmentation_type)
    
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint)    
    # Keypoint Normalization
    keypoint = keypoint_normalization(keypoint)

    return keypoint

def remove_keypoints(keypoint):
    pose_keep = torch.tensor([i for i in range(18) if i not in {7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 17}], dtype=torch.long)

    pose    = torch.tensor(keypoint[:,pose_keep,:])
    face    = torch.tensor(keypoint[:,28:92,:])
    left_h  = torch.tensor(keypoint[:,93:113,:])
    right_h = torch.tensor(keypoint[:,113:134,:])

    keypoints = torch.cat([pose, face, left_h, right_h], dim=1)
    return keypoints