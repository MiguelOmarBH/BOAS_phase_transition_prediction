"""Download the pinned BOAS files and verify their SHA-256 checksums."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(record: dict, root: Path) -> Path:
    target = root / record["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and target.stat().st_size == record["size"]
        and checksum(target) == record["checksum"]
    ):
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    errors = []
    for url in (record["bytes_url"], record["object_url"]):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes=0-{record['size'] - 1}"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size != record["size"]:
                raise IOError("Size mismatch")
            if checksum(temporary) != record["checksum"]:
                raise IOError("SHA-256 mismatch")
            temporary.replace(target)
            return target
        except (OSError, ValueError) as error:
            errors.append(str(error))
    temporary.unlink(missing_ok=True)
    raise IOError(f"{record['path']}: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    records = json.loads((ROOT / "data/metadata/source_manifest.json").read_text())
    if args.metadata_only:
        records = [r for r in records if not r["path"].endswith(".edf")]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for record in records:
            target_root = ROOT / ("data/raw" if record["path"].endswith(".edf") else "data/source")
            futures[executor.submit(download, record, target_root)] = record
        for number, future in enumerate(as_completed(futures), 1):
            print(f"[{number}/{len(records)}] {future.result().relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
