#!/usr/bin/env python3
"""List evaluation_sessions from the RoboArena Hugging Face datadump."""

import argparse
from urllib.parse import urlparse

HF_SOURCE = "https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026/tree/main/evaluation_sessions"


def _parse_hf_url_to_repo(url: str) -> str:
    """
    Parse e.g.
    https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026/tree/main/evaluation_sessions
    -> RoboArena/DataDump_02-03-2026
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Unrecognized Hugging Face dataset URL: {url}")
    return f"{parts[1]}/{parts[2]}"


def _list_hf_dataset_dir(source: str, path_in_repo: str, revision: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hugging Face listing. "
            "Install with: pip install huggingface_hub"
        ) from exc

    if source.startswith("https://huggingface.co/"):
        repo_id = _parse_hf_url_to_repo(source)
    else:
        repo_id = source

    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="dataset",
            revision=revision,
            recursive=False,
        )
    )
    print(len(entries))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List evaluation_sessions from the hardcoded Hugging Face source."
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default="evaluation_sessions",
        help="Directory to list inside the HF dataset repo.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face revision/branch/tag for remote listings.",
    )
    args = parser.parse_args()
    source = HF_SOURCE
    _list_hf_dataset_dir(
        source=source,
        path_in_repo=args.dirname,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
