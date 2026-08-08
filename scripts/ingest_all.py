"""CLI to run one or both ingestion pipelines against data/raw/.

    uv run python scripts/ingest_all.py --pipeline both --rebuild

Ingestion is a deliberately separate, offline step from the Streamlit app: extraction
and embedding happen here, once, ahead of time. The app only does retrieval and answer
generation at query time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fyp_demo import config
from fyp_demo.ingest_akshara import ingest_akshara
from fyp_demo.ingest_baseline import ingest_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["akshara", "baseline", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection(s) before ingesting",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.pipeline in ("baseline", "both"):
        n = ingest_baseline(args.data_dir, rebuild=args.rebuild)
        print(f"[baseline] ingested {n} chunks")

    if args.pipeline in ("akshara", "both"):
        n = ingest_akshara(args.data_dir, rebuild=args.rebuild)
        print(f"[akshara] ingested {n} chunks")


if __name__ == "__main__":
    main()
