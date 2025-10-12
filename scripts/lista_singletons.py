#!/usr/bin/env python3
"""
Analiza un dataset HDF5 de señas y reporta:
  1) Etiquetas desestimadas por contener espacios o guiones.
  2) Etiquetas con caracteres fuera del alfabeto español permitido.
  3) Etiquetas con un solo muestreo (singletons) tras los filtros anteriores.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import h5py

# Caracteres permitidos por defecto (minúsculas/ mayúsculas se agregan automáticamente)
DEFAULT_ALLOWED_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "áéíóúüñ"
)
DEFAULT_ALLOWED_EXTRA = "¡¿"
DEFAULT_PUNCT_CHARS = "!?¿¡.,;:…"


@dataclass
class LabelInfo:
    count: int = 0
    datasets: set[str] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listado de labels poco frecuentes o con caracteres fuera del alfabeto español."
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Ruta al archivo HDF5 (por ejemplo /run/media/.../dataset_v9.hdf5).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dataset1", "dataset3", "dataset5", "dataset7"],
        help="Conjuntos a incluir (por defecto: dataset1, dataset3, dataset5, dataset7).",
    )
    parser.add_argument(
        "--allow-chars",
        default=DEFAULT_ALLOWED_CHARS,
        help="Cadena con caracteres básicos permitidos (solo minúsculas).",
    )
    parser.add_argument(
        "--allow-extra",
        default=DEFAULT_ALLOWED_EXTRA,
        help="Caracteres adicionales permitidos (por ejemplo '¡¿').",
    )
    parser.add_argument(
        "--allow-digits",
        action="store_true",
        help="Permite dígitos 0-9.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Umbral mínimo de apariciones para considerar una etiqueta (por defecto 1).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Longitud mínima permitida para una etiqueta (después del filtrado).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Longitud máxima permitida (0 desactiva el límite).",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=1,
        help="Número mínimo de datasets distintos en los que debe aparecer una etiqueta.",
    )
    parser.add_argument(
        "--punct-chars",
        default=DEFAULT_PUNCT_CHARS,
        help="Caracteres de puntuación que disparan la bandera de ruido.",
    )
    parser.add_argument(
        "--json",
        help="Si se especifica, guarda resultados en este archivo JSON además de mostrarlos.",
    )
    return parser.parse_args()


def build_allowed_set(args: argparse.Namespace) -> set[str]:
    allowed = set()
    allowed.update(args.allow_chars)
    allowed.update(args.allow_chars.upper())
    allowed.update(args.allow_extra)
    allowed.update(args.allow_extra.upper())

    if args.allow_digits:
        allowed.update("0123456789")

    return allowed


def label_has_only_allowed_chars(label: str, allowed_chars: set[str]) -> bool:
    norm = unicodedata.normalize("NFC", label)
    return all(ch in allowed_chars for ch in norm)


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def main() -> None:
    args = parse_args()
    allowed_datasets = set(args.datasets)
    allowed_chars = build_allowed_set(args)

    label_counts = Counter()
    space_or_hyphen_counts = Counter()
    invalid_char_counts = Counter()
    label_info = defaultdict(LabelInfo)

    total_samples = 0
    filtered_samples = 0

    with h5py.File(args.h5, "r") as h5_file:
        for dataset_name in sorted(h5_file.keys()):
            if dataset_name not in allowed_datasets:
                continue

            label_group = h5_file[dataset_name]["labels"]
            for clip_id in label_group:
                raw = label_group[clip_id][:][0]
                label = raw.decode("utf-8").strip()
                total_samples += 1

                if " " in label or "-" in label:
                    space_or_hyphen_counts[label] += 1
                    continue

                if not label_has_only_allowed_chars(label.lower(), allowed_chars):
                    invalid_char_counts[label] += 1
                    continue

                label_counts[label] += 1
                filtered_samples += 1
                info = label_info[label]
                info.count += 1
                info.datasets.add(dataset_name)

    singleton_labels = sorted(label for label, count in label_counts.items() if count == 1)
    unique_filtered = len(label_counts)
    singleton_pct = (len(singleton_labels) / unique_filtered * 100) if unique_filtered else 0.0

    min_freq_labels = sorted(
        label for label, count in label_counts.items() if count < max(args.min_frequency, 1)
    )
    short_labels = sorted(
        label
        for label in label_counts
        if len(label) < max(args.min_length, 1)
    )
    long_labels = sorted(
        label
        for label in label_counts
        if args.max_length and len(label) > args.max_length
    )
    dataset_coverage_labels = sorted(
        label
        for label, info in label_info.items()
        if len(info.datasets) < max(args.min_datasets, 1)
    )
    punct_labels = sorted(
        label
        for label in label_counts
        if any(ch in args.punct_chars for ch in label)
    )

    normalized_map = defaultdict(list)
    for label in label_counts:
        normalized = strip_accents(label.lower())
        normalized_map[normalized].append(label)

    normalized_collisions = []
    for normalized, variants in sorted(normalized_map.items()):
        if len(variants) <= 1:
            continue
        variant_info = [
            {"label": variant, "count": label_counts[variant], "datasets": sorted(label_info[variant].datasets)}
            for variant in sorted(variants)
        ]
        variant_info.sort(key=lambda item: item["count"], reverse=True)
        normalized_collisions.append(
            {
                "normalized": normalized,
                "variants": variant_info,
            }
        )

    summary = {
        "total_samples": total_samples,
        "filtered_samples": filtered_samples,
        "unique_labels_filtered": unique_filtered,
        "singleton_labels_count": len(singleton_labels),
        "singleton_labels_percentage": singleton_pct,
        "space_or_hyphen_labels_count": len(space_or_hyphen_counts),
        "space_or_hyphen_samples": sum(space_or_hyphen_counts.values()),
        "invalid_spanish_labels_count": len(invalid_char_counts),
        "invalid_spanish_samples": sum(invalid_char_counts.values()),
        "singleton_labels": singleton_labels,
        "invalid_spanish_labels": sorted(invalid_char_counts),
        "low_frequency_labels": min_freq_labels,
        "short_labels": short_labels,
        "long_labels": long_labels,
        "labels_below_dataset_threshold": dataset_coverage_labels,
        "punctuation_labels": punct_labels,
        "normalized_collisions": normalized_collisions,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"Resumen guardado en {args.json!r}")


if __name__ == "__main__":
    main()
