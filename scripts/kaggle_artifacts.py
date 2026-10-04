"""Small Kaggle-only artifact backup/restore helper.

This script is operational tooling, not part of the research model.

Typical Kaggle workflow:

    # First ever backup:
    # Creates both the local artifact store and Kaggle Dataset.
    python scripts/kaggle_artifacts.py backup \
        --handle USERNAME/diffusion-portfolio \
        --experiment phase0/neural_cde_rk4x4 \
        --source /kaggle/working/phase0_cde_rk4x4

    # At the start of later Kaggle sessions:
    python scripts/kaggle_artifacts.py restore \
        --handle USERNAME/diffusion-portfolio

    # Then back up new/updated experiment artifacts:
    python scripts/kaggle_artifacts.py backup \
        --handle USERNAME/diffusion-portfolio \
        --experiment phase1/example \
        --source /kaggle/working/example

The local artifact mirror defaults to:

    /kaggle/working/diffusion-portfolio-artifacts

Training/evaluation code does not depend on this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path(
    "/kaggle/working/"
    "diffusion-portfolio-artifacts"
)

MARKER_NAME = ".artifact_store.json"


def _kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for this helper. "
            "It is normally available in Kaggle notebooks."
        ) from exc

    return kagglehub


def _now_utc() -> str:
    return (
        dt.datetime.now(
            dt.timezone.utc
        )
        .isoformat()
    )


def _marker_path(
    root: Path,
) -> Path:
    return (
        root
        / MARKER_NAME
    )


def _write_marker(
    root: Path,
    handle: str,
) -> None:
    payload = {
        "schema_version": 1,
        "handle": handle,
        "created_at_utc": (
            _now_utc()
        ),
    }

    with _marker_path(
        root
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def _validate_store(
    root: Path,
    handle: str,
) -> None:
    marker_path = (
        _marker_path(
            root
        )
    )

    if not marker_path.exists():
        raise RuntimeError(
            f"Artifact-store marker missing: "
            f"{marker_path}. "
            "Run restore first."
        )

    with marker_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        marker = json.load(
            file
        )

    if marker.get(
        "handle"
    ) != handle:
        raise RuntimeError(
            "Artifact store belongs to "
            f"{marker.get('handle')!r}, "
            f"not {handle!r}."
        )


def _validate_experiment(
    experiment: str,
) -> Path:
    path = Path(
        experiment
    )

    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(
            "experiment must be a safe "
            "relative path"
        )

    return path


def _git_commit() -> str | None:
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return None

    return (
        result.stdout.strip()
    )


def restore_store(
    *,
    handle: str,
    root: Path,
    force: bool,
) -> None:
    if root.exists():
        if any(
            root.iterdir()
        ) and not force:
            raise RuntimeError(
                f"{root} is not empty. "
                "Use --force only if you want "
                "to replace it with the latest "
                "Kaggle dataset version."
            )

        shutil.rmtree(
            root
        )

    root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kagglehub = (
        _kagglehub()
    )

    kagglehub.dataset_download(
        handle,
        output_dir=str(
            root
        ),
        force_download=True,
    )

    _validate_store(
        root,
        handle,
    )

    print(
        "Restored artifact store:"
    )

    print(
        "  handle:",
        handle,
    )

    print(
        "  local mirror:",
        root,
    )


def backup_experiment(
    *,
    handle: str,
    root: Path,
    experiment: str,
    source: Path,
    notes: str | None,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            source
        )

    marker_path = (
        _marker_path(
            root
        )
    )

    if not root.exists():
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_marker(
            root,
            handle,
        )

        print(
            "Creating new artifact store:"
        )

        print(
            "  handle:",
            handle,
        )

    elif not marker_path.exists():
        if any(
            root.iterdir()
        ):
            raise RuntimeError(
                f"{root} is non-empty but is not "
                "a restored artifact store. "
                "Run restore before backup."
            )

        _write_marker(
            root,
            handle,
        )

    else:
        _validate_store(
            root,
            handle,
        )

    relative_destination = (
        _validate_experiment(
            experiment
        )
    )

    destination = (
        root
        / relative_destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        shutil.rmtree(
            destination
        )

    shutil.copytree(
        source,
        destination,
    )

    metadata = {
        "experiment": experiment,
        "backed_up_at_utc": (
            _now_utc()
        ),
        "git_commit": (
            _git_commit()
        ),
    }

    with (
        destination
        / "artifact_meta.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    kagglehub = (
        _kagglehub()
    )

    version_notes = (
        notes
        if notes
        else f"Backup {experiment}"
    )

    kagglehub.dataset_upload(
        handle,
        str(root),
        version_notes=(
            version_notes
        ),
        ignore_patterns=[
            ".complete",
        ],
    )

    print(
        "Backed up experiment:"
    )

    print(
        "  experiment:",
        experiment,
    )

    print(
        "  source:",
        source,
    )

    print(
        "  stored at:",
        destination,
    )

    print(
        "  dataset:",
        handle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Kaggle artifact "
            "backup/restore helper."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Local mirror of the central "
            "artifact dataset."
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )


    restore_parser = (
        subparsers.add_parser(
            "restore"
        )
    )

    restore_parser.add_argument(
        "--handle",
        required=True,
    )

    restore_parser.add_argument(
        "--force",
        action="store_true",
    )

    backup_parser = (
        subparsers.add_parser(
            "backup"
        )
    )

    backup_parser.add_argument(
        "--handle",
        required=True,
    )

    backup_parser.add_argument(
        "--experiment",
        required=True,
    )

    backup_parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    backup_parser.add_argument(
        "--notes",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "restore":
        restore_store(
            handle=args.handle,
            root=args.root,
            force=args.force,
        )

    elif args.command == "backup":
        backup_experiment(
            handle=args.handle,
            root=args.root,
            experiment=(
                args.experiment
            ),
            source=args.source,
            notes=args.notes,
        )

    else:
        raise RuntimeError(
            f"Unknown command: "
            f"{args.command}"
        )


if __name__ == "__main__":
    main()