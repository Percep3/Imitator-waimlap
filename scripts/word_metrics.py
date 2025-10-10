#!/usr/bin/env python3
"""
Evalúa predicciones palabra por palabra usando métricas enfocadas al dominio:
- Exact match (case-insensitive) considerando múltiples referencias por muestra.
- Distancia de Levenshtein mínima frente a las referencias.
- Similitud normalizada 1 - (distancia / longitud_máxima).

Para aprovechar el flujo existente:
- Reutiliza el caché HDF5 generado por benchchrF (generándolo si es necesario).
- Convierte embeddings a texto con el mismo LLM del módulo BLEU.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

# Asegura que podamos importar módulos del proyecto y reutilizar helpers del benchmark.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Evita bloqueos HDF5 cuando se reabre inmediatamente el archivo recién escrito.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from src.mslm.benchmark import BLEU as bleu_module  # noqa: E402
import benchchrF  # type: ignore  # noqa: E402


@dataclass
class SampleResult:
    prediction: str
    references: List[str]
    best_distance: int
    best_similarity: float
    is_exact_match: bool


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def _levenshtein(a: str, b: str) -> int:
    """
    Implementación iterativa O(len(a) * len(b)) que usa dos filas para minimizar memoria.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def _gather_predictions(
    version: str,
    checkpoint: str,
    epoch: int,
    *,
    cache_path: str,
    mask_embds_is_padding: bool,
) -> Iterable[Tuple[str, List[str]]]:
    """
    Garantiza que exista un caché válido y devuelve pares (predicción, referencias).
    """
    h5_file, _, model_cfg = bleu_module.load_config()
    dataloader, id_to_label, _ = bleu_module.load_dataset(h5_file, model_cfg.get("n_keypoints", 111))

    if not benchchrF._is_cache_complete(cache_path):
        model = bleu_module.load_model(model_cfg, version, checkpoint, epoch)
        model.eval().to(bleu_module.device)
        benchchrF._generate_cache(
            dataloader,
            id_to_label,
            model,
            cache_path,
            mask_embds_is_padding=mask_embds_is_padding,
        )

    tokenizer, all_embeddings = bleu_module.load_llm()

    for embed_array, hyps in benchchrF._iter_cached_samples(cache_path):
        embed_pred = torch.from_numpy(embed_array).to(dtype=torch.float32)
        pred_text = bleu_module.embeddings_to_text(embed_pred, all_embeddings, tokenizer) or ""
        yield pred_text, list(hyps)


def evaluate_samples(samples: Iterable[Tuple[str, Sequence[str]]]) -> List[SampleResult]:
    results: List[SampleResult] = []

    for prediction, refs in samples:
        pred_norm = _normalize_text(prediction)
        refs_norm = [_normalize_text(ref) for ref in refs if ref]
        if not refs_norm:
            continue

        best_distance = np.inf
        best_similarity = -np.inf
        exact_match = False

        for ref_norm in refs_norm:
            distance = _levenshtein(pred_norm, ref_norm)
            max_len = max(len(pred_norm), len(ref_norm), 1)
            similarity = 1.0 - distance / max_len

            if distance < best_distance:
                best_distance = distance
                best_similarity = similarity

            if distance == 0:
                exact_match = True
                best_distance = 0
                best_similarity = 1.0
                break

        results.append(
            SampleResult(
                prediction=prediction.strip(),
                references=[ref.strip() for ref in refs],
                best_distance=int(best_distance),
                best_similarity=float(best_similarity),
                is_exact_match=exact_match,
            )
        )

    return results


def summarize_results(results: Sequence[SampleResult], top_k: int = 10) -> None:
    total = len(results)
    if total == 0:
        print("No se encontraron muestras para evaluar.")
        return

    exact_matches = sum(result.is_exact_match for result in results)
    avg_distance = sum(result.best_distance for result in results) / total
    avg_similarity = sum(result.best_similarity for result in results) / total

    print(f"Total samples evaluated: {total}")
    print(f"Exact match rate: {exact_matches / total * 100:.2f}% ({exact_matches}/{total})")
    print(f"Average Levenshtein distance: {avg_distance:.3f}")
    print(f"Average normalized similarity: {avg_similarity * 100:.2f}%")

    if top_k <= 0:
        return

    print("\nTop mispredictions (sorted by highest distance):")
    worst = sorted(results, key=lambda r: (r.best_distance, r.best_similarity), reverse=True)[:top_k]
    for idx, sample in enumerate(worst, start=1):
        refs_joined = " | ".join(sample.references)
        print(f"{idx:02d}. pred='{sample.prediction}' | refs='{refs_joined}' | "
              f"dist={sample.best_distance} | sim={sample.best_similarity * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute word-level metrics (exact match & Levenshtein) for imitator predictions."
    )
    parser.add_argument("--version", type=str, required=True, help="Model version to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to evaluate.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the checkpoint to evaluate.")
    parser.add_argument("--cache-path", type=str, default="bleu_imitator_embeds.h5", help="Path to HDF5 cache.")
    parser.add_argument(
        "--mask-embds-is-valid",
        action="store_true",
        help="Usa esta opción si mask_embds == True significa token VÁLIDO (por defecto se asume True == padding).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Número de ejemplos con mayor distancia a mostrar (0 para omitir).",
    )

    args = parser.parse_args()
    mask_is_padding = not args.mask_embds_is_valid

    samples = _gather_predictions(
        args.version,
        args.checkpoint,
        args.epoch,
        cache_path=args.cache_path,
        mask_embds_is_padding=mask_is_padding,
    )
    results = evaluate_samples(samples)
    summarize_results(results, top_k=args.top_k)


if __name__ == "__main__":
    main()
