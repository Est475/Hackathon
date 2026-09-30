"""Conversion de la timeline en tableau numpy structuré."""

from __future__ import annotations

import numpy as np

COLONNES = ("esperance", "q05", "q25", "q50", "q75", "q95", "p_decouvert")

DTYPE_TIMELINE = np.dtype([("date", "datetime64[D]")] + [(c, "f8") for c in COLONNES])


def timeline_numpy(sortie: dict) -> np.ndarray:
    """Timeline de predict() → tableau structuré de forme (D+1,), une ligne par jour.

    Accès par colonne : t["date"], t["esperance"], t["q05"], …, t["p_decouvert"].
    """
    timeline = sortie["timeline"]
    t = np.empty(len(timeline), dtype=DTYPE_TIMELINE)
    t["date"] = [j["date"] for j in timeline]
    for c in COLONNES:
        t[c] = [j[c] for j in timeline]
    return t
