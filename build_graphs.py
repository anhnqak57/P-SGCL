from __future__ import annotations

import argparse
import sys

from psgcl_core.data import supported_dataset_names
from psgcl_core.preprocess import build_graphs




for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build new adaptive-percentile sample graphs.")
    parser.add_argument("--dataset", choices=supported_dataset_names(), required=True)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(build_graphs(args.dataset, args.data_dir, args.output_dir))


if __name__ == "__main__":
    main()
