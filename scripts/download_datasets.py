#!/usr/bin/env python3
"""Download versioned Qasper/SciFact source files with SHA256 manifests."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "qasper": {
        "train": {
            "url": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
            "member": "qasper-train-v0.3.json",
            "filename": "train.json",
        },
        "dev": {
            "url": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
            "member": "qasper-dev-v0.3.json",
            "filename": "dev.json",
        },
        "test": {
            "url": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
            "member": "qasper-test-v0.3.json",
            "filename": "test.json",
        },
    },
    "scifact": {
        "corpus": {
            "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "member": "data/corpus.jsonl",
            "filename": "corpus.jsonl",
        },
        "claims_train": {
            "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "member": "data/claims_train.jsonl",
            "filename": "claims_train.jsonl",
        },
        "claims_dev": {
            "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "member": "data/claims_dev.jsonl",
            "filename": "claims_dev.jsonl",
        },
        "claims_test": {
            "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "member": "data/claims_test.jsonl",
            "filename": "claims_test.jsonl",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "GrantScout/0.1"})
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def extract_member(archive: Path, member: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            info = handle.getmember(member)
            source = handle.extractfile(info)
            if source is None:
                raise RuntimeError(f"Archive member is not a file: {member}")
            with partial.open("wb") as output:
                shutil.copyfileobj(source, output)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", default="all", help="Split name or all.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = DATASETS[args.dataset]
    if args.split != "all":
        if args.split not in selected:
            raise SystemExit(f"Unknown split {args.split!r} for {args.dataset}")
        selected = {args.split: selected[args.split]}
    dataset_dir = args.output_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archives: dict[str, Path] = {}
    manifest = {
        "dataset": args.dataset,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for name, spec in selected.items():
        archive = archives.get(spec["url"])
        if archive is None:
            archive = dataset_dir / Path(spec["url"]).name
            if not archive.exists():
                print(f"Downloading {spec['url']} -> {archive}")
                download(spec["url"], archive, timeout=args.timeout)
            archives[spec["url"]] = archive
        target = dataset_dir / spec["filename"]
        print(f"Extracting {spec['member']} -> {target}")
        extract_member(archive, spec["member"], target)
        manifest["files"].append(
            {
                "name": name,
                "url": spec["url"],
                "archive": str(archive.relative_to(args.output_dir)),
                "member": spec["member"],
                "path": str(target.relative_to(args.output_dir)),
                "sha256": sha256(target),
            }
        )
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
