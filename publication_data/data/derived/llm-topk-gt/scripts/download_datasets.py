#!/usr/bin/env python3
"""Download BEIR datasets for the LLM Top-K Ground Truth project.

This script downloads specified BEIR datasets to the data/raw directory.

Usage
-----
    # Download specific datasets
    python scripts/download_datasets.py --datasets scifact scidocs

    # Download all supported datasets
    python scripts/download_datasets.py --all

    # Force re-download
    python scripts/download_datasets.py --datasets scifact --force

    # Custom data directory
    python scripts/download_datasets.py --datasets scifact --data-dir /path/to/data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    download_datasets,
    list_available_datasets,
    load_dataset,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Parameters
    ----------
    verbose : bool
        If True, use DEBUG level. Otherwise INFO.
    """
    level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download BEIR datasets for the LLM Top-K Ground Truth project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download SciFact dataset
    python scripts/download_datasets.py --datasets scifact

    # Download multiple datasets
    python scripts/download_datasets.py --datasets scifact scidocs

    # Download all supported datasets
    python scripts/download_datasets.py --all

    # Force re-download even if exists
    python scripts/download_datasets.py --datasets scifact --force
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="Dataset names to download (e.g., scifact scidocs)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported datasets",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory (default: ./data)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded datasets by loading them",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args: argparse.Namespace = parse_args()
    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    # Handle --list flag
    if args.list:
        available: set[str] = list_available_datasets()
        print("Available datasets:")
        for name in sorted(available):
            print(f"  - {name}")
        return 0

    # Determine which datasets to download
    datasets_to_download: list[str]

    if args.all:
        datasets_to_download = sorted(list_available_datasets())
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        logger.error("No datasets specified. Use --datasets or --all")
        return 1

    # Validate dataset names
    available: set[str] = list_available_datasets()
    invalid: list[str] = [d for d in datasets_to_download if d not in available]

    if invalid:
        logger.error(
            f"Invalid dataset names: {invalid}. "
            f"Available: {sorted(available)}"
        )
        return 1

    # Create data directory
    data_dir: Path = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Datasets to download: {datasets_to_download}")

    # Download datasets
    try:
        results: dict[str, Path] = download_datasets(
            datasets_to_download,
            data_dir,
            force=args.force,
        )

        print("\nDownload complete!")
        print("-" * 40)
        for name, path in results.items():
            print(f"  {name}: {path}")

    except Exception as e:
        logger.exception(f"Download failed: {e}")
        return 1

    # Verify if requested
    if args.verify:
        print("\nVerifying datasets...")
        print("-" * 40)

        for name in datasets_to_download:
            try:
                dataset = load_dataset(name, data_dir, download_if_missing=False)
                print(
                    f"  {name}: OK "
                    f"({dataset.info.num_documents} docs, "
                    f"{dataset.info.num_queries} queries, "
                    f"{dataset.info.num_qrels} qrels)"
                )
            except Exception as e:
                logger.error(f"  {name}: FAILED - {e}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())