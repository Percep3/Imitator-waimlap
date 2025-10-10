#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evalúa un modelo usando chrF++ (caracteres + n-gramas de palabras) con sacrebleu.
- Genera o reutiliza un caché HDF5 con predicciones (embeddings -> texto) y referencias.
- Soporta batch_size > 1.
- Corrige criterios de caché incompleto.
- Aplica normalización ligera y robustez en refs.
- chrF++ explícito: word_order=2, char_order=6, beta=2, lowercase=True.
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm

# Ruta raíz del proyecto (ajústala si cambia estructura)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mslm.benchmark import BLEU as bleu_module  # noqa: E402


# ==========================================================
# 1. Verificación de caché
# ==========================================================
def _is_cache_complete(cache_path: str) -> bool:
    """El caché se considera completo si el atributo 'processed_samples' coincide con 'expected_valid_samples'."""
    if not os.path.exists(cache_path):
        return False
    with h5py.File(cache_path, "r") as h5f:
        processed = int(h5f.attrs.get("processed_samples", 0))
        expected = h5f.attrs.get("expected_valid_samples", -1)
        return expected != -1 and processed == expected


# ==========================================================
# 2. Iterador de muestras cacheadas
# ==========================================================
def _iter_cached_samples(cache_path: str) -> Iterable[Tuple[np.ndarray, List[str]]]:
    """Itera sobre el HDF5 y devuelve pares (embed_pred, referencias)."""
    with h5py.File(cache_path, "r") as h5f:
        samples_group = h5f["samples"]
        for key in sorted(samples_group.keys()):
            sample_group = samples_group[key]
            hyps_data = sample_group["hyps"][()]

            if isinstance(hyps_data, np.ndarray):
                hyps_list = hyps_data.tolist()
            else:
                hyps_list = list(hyps_data)

            hyps_list = [
                h.decode("utf-8") if isinstance(h, (bytes, np.bytes_)) else str(h)
                for h in hyps_list
            ]
            yield sample_group["embed_pred"][()], hyps_list


# ==========================================================
# 3. Normalización de referencias
# ==========================================================
def _prepare_reference_streams(references: Sequence[Sequence[str]]) -> List[List[str]]:
    """Asegura que todas las muestras tengan el mismo número de referencias."""
    if not references:
        raise ValueError("No references available to compute chrF++.")

    max_refs = max(len(ref_list) for ref_list in references)
    ref_streams: List[List[str]] = [[] for _ in range(max_refs)]

    for ref_list in references:
        if not ref_list:
            raise ValueError("Found a sample with empty reference list.")
        fallback = ref_list[0]
        for idx in range(max_refs):
            variant = ref_list[idx] if idx < len(ref_list) else fallback
            ref_streams[idx].append(variant.strip())

    return ref_streams


# ==========================================================
# 4. Cálculo de chrF++
# ==========================================================
def _compute_chrf(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    lowercase: bool = True,
    beta: float = 2.0,
    char_order: int = 6,
    word_order: int = 2,
) -> float:
    """
    Calcula chrF o chrF++:
    - word_order=2 → activa n-gramas de palabras → chrF++.
    - lowercase=True → hace la métrica insensible a mayúsculas/minúsculas.
    """
    metric = CHRF(beta=beta, char_order=char_order, word_order=word_order, lowercase=lowercase)
    preds_norm = [p.strip() if isinstance(p, str) else "" for p in predictions]
    ref_streams = _prepare_reference_streams(references)
    return metric.corpus_score(preds_norm, ref_streams).score


# ==========================================================
# 5. Generación del caché
# ==========================================================
def _generate_cache(
    dataloader,
    id_to_label: Dict[int, str],
    model: torch.nn.Module,
    cache_path: str,
    *,
    mask_embds_is_padding: bool = True,
) -> None:
    """
    Ejecuta el modelo y guarda las predicciones y referencias en un HDF5.
    TODO: valida que:
      - mask_embds == True → padding (o ajusta el flag si es al revés).
      - bleu_module.HYPS[idx] contenga strings válidos.
      - embeddings_to_text devuelva texto coherente.
    """
    string_dtype = h5py.string_dtype(encoding="utf-8")
    device = bleu_module.device
    valid_count = 0

    with h5py.File(cache_path, "w") as h5f:
        samples_group = h5f.create_group("samples")

        for batch in tqdm(dataloader, desc="Processing samples"):
            keypoints, mask_data, mask_embds, label_id = batch

            keypoints = keypoints.to(device=device, dtype=torch.float32)
            mask_data = mask_data.to(device=device)
            mask_embds = mask_embds.to(device=device)

            with torch.inference_mode():
                sign_embed, _ = model(keypoints, mask_data)
                B = sign_embed.shape[0]

                for b in range(B):
                    label_text = id_to_label[int(label_id[b])]
                    idx = bleu_module.get_idx_hyps(label_text)
                    if idx == -1:
                        continue

                    emb_mask = ~mask_embds[b] if mask_embds_is_padding else mask_embds[b]
                    seq_len = int(emb_mask.sum().item())
                    if seq_len <= 0:
                        continue

                    max_available = sign_embed[b].shape[0]
                    seq_len = min(seq_len, max_available)
                    pred_embeds = sign_embed[b][:seq_len].contiguous()
                    hyps = bleu_module.HYPS[idx]
                    if not hyps:
                        continue

                    sample_group = samples_group.create_group(f"{valid_count:06d}")
                    sample_group.attrs["label"] = label_text
                    sample_group.create_dataset(
                        "embed_pred",
                        data=pred_embeds.detach().cpu().numpy(),
                        compression="gzip",
                    )
                    ds = sample_group.create_dataset("hyps", (len(hyps),), dtype=string_dtype)
                    ds[:] = hyps

                    valid_count += 1

        h5f.attrs["processed_samples"] = valid_count
        h5f.attrs["expected_valid_samples"] = valid_count


# ==========================================================
# 6. Pipeline principal
# ==========================================================
def run(
    version: str,
    checkpoint: str,
    epoch: int,
    *,
    cache_path: str = "bleu_imitator_embeds.h5",
    mask_embds_is_padding: bool = True,
) -> float:
    """Carga datos y modelo, genera caché si no existe y calcula chrF++."""
    h5_file, _, model_cfg = bleu_module.load_config()
    dataloader, id_to_label, _ = bleu_module.load_dataset(h5_file, model_cfg.get("n_keypoints", 111))

    if not _is_cache_complete(cache_path):
        model = bleu_module.load_model(model_cfg, version, checkpoint, epoch)
        model.eval().to(bleu_module.device)
        _generate_cache(dataloader, id_to_label, model, cache_path, mask_embds_is_padding=mask_embds_is_padding)

    tokenizer, all_embeddings = bleu_module.load_llm()

    predictions: List[str] = []
    references: List[List[str]] = []

    for embed_array, hyps in _iter_cached_samples(cache_path):
        embed_pred = torch.from_numpy(embed_array).to(dtype=torch.float32)
        pred_text = bleu_module.embeddings_to_text(embed_pred, all_embeddings, tokenizer) or ""
        predictions.append(pred_text)
        references.append(hyps)

    if not predictions:
        raise RuntimeError("No se generaron predicciones. Revisa el pipeline y los filtros.")

    score = _compute_chrf(predictions, references)
    print(f"chrF++ score: {score:.2f}")
    return score


# ==========================================================
# 7. CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using chrF++ score.")
    parser.add_argument("--version", type=str, required=True, help="Model version to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to evaluate.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the checkpoint to evaluate.")
    parser.add_argument("--cache-path", type=str, default="bleu_imitator_embeds.h5", help="Path to HDF5 cache.")
    parser.add_argument(
        "--mask-embds-is-valid",
        action="store_true",
        help="Usa esta opción si mask_embds == True significa token VÁLIDO (por defecto se asume True == padding).",
    )

    args = parser.parse_args()
    mask_is_padding = not args.mask_embds_is_valid

    run(
        args.version,
        args.checkpoint,
        args.epoch,
        cache_path=args.cache_path,
        mask_embds_is_padding=mask_is_padding,
    )
