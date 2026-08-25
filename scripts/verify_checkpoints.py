"""Check every checkpoint under a directory, and print a line per file to diff by hand.

A torch checkpoint is a zip archive, so integrity is ``zipfile``'s question and needs
neither torch nor the project environment -- which is the point: this runs on the remote
machine with nothing installed. A truncated transfer loses the archive's central
directory, which lives at the *end* of the file, so a partial copy has a valid header and
fails to open.

    python3 scripts/verify_checkpoints.py runs > local.txt      # here
    ssh remote 'python3 -' < scripts/verify_checkpoints.py      # there, reading runs/

Diff the two. A file that is ok there and bad here was truncated in transit; one that is
bad on both was written badly and has to be retrained.

The hash covers the bytes, so two ok lines with the same digest are the same checkpoint.
Pass --quick to skip hashing when the sizes alone settle it.
"""

import hashlib
import sys
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def main() -> None:
    arguments = [a for a in sys.argv[1:] if not a.startswith("-")]
    quick = "--quick" in sys.argv

    root = Path(arguments[0] if arguments else "runs")
    files = sorted(root.rglob("*.ckpt"))
    if not files:
        raise SystemExit(f"no *.ckpt under {root}")

    bad = 0
    for path in files:
        try:
            with zipfile.ZipFile(path) as archive:
                # Reading the directory catches truncation; testzip catches a corrupt
                # member, which is the rarer case but costs one pass over the file.
                broken = archive.testzip()
            status = "ok" if broken is None else f"BAD:{broken}"
        except Exception as error:
            status = f"BAD:{type(error).__name__}"

        bad += not status.startswith("ok")
        stamp = "-" * 16 if quick else digest(path)
        print(f"{status:<24} {path.stat().st_size:>12} {stamp}  {path}")

    print(f"\n{bad}/{len(files)} bad under {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
