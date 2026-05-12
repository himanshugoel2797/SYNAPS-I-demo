"""
One-shot test of segmentation_function() from receiver_3d.py.

Reads 'hxn/processed/reconstructions/fista_recon1' from Tiled and passes it
directly to segmentation_function(), which handles the full SAM3 pipeline and
writes results to 'hxn/processed/segmentations/fista_recon1'.

Environment variables:
  API_KEY   – Tiled API key (required)
  SAM_CKPT  – path to SAM3 checkpoint (optional, falls back to DEFAULT_CKPT)
"""

import os

import numpy as np
from tiled.client import from_uri

# Importing receiver_3d runs its module-level setup (writer_client, executor).
# API_KEY must be set before this import.
from receiver_3d import segmentation_function, URI_IN

DATASET = "fista_recon1"
# path_parts mimics what run_segmentation() passes; [-2] must be the dataset name.
PATH_PARTS = ("hxn", "processed", "reconstructions", DATASET, DATASET)


def main() -> None:
    api_key = os.getenv("API_KEY")

    print(f"Reading '{DATASET}' from {URI_IN} …")
    reader = from_uri(URI_IN, api_key=api_key)
    data = reader[DATASET]
    print(f"  shape: {np.asarray(data).shape}  dtype: {np.asarray(data).dtype}")

    segmentation_function(data=data, metadata={}, path_parts=PATH_PARTS)


if __name__ == "__main__":
    main()
