"""Fetch and verify unchanged official v1.0.0 data using only stdlib."""
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/assignment2-data-v1.0.0.zip"
ARCHIVE_SHA256 = "59ab05aa2f75a9d589d5a26a65d10bdde1f7f88562af163534f29905002c7a20"
FILES = {"data/train.bin", "data/val.bin", "data/tokenizer.json", "data/manifest.json",
         "data/LICENSES.md", "data/CDLA-Sharing-1.0.html"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name, info in manifest["files"].items():
        path = root / "data" / name
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Data verification failed: {name}")


def main():
    (ROOT / "work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a2-data-", dir=ROOT / "work") as temp:
        stage = Path(temp)
        archive = stage / "data.zip"
        print("Downloading official fixed data v1.0.0 ...")
        urllib.request.urlretrieve(URL, archive)
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError("Archive SHA-256 mismatch; nothing was installed")
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(FILES) or set(z.namelist()) != FILES:
                raise ValueError("Unexpected archive entries; nothing was installed")
            z.extractall(stage)
        verify(stage)
        for name in FILES:
            dest = ROOT / name
            if dest.exists() and (not dest.is_file() or digest(dest) != digest(stage / name)):
                raise FileExistsError(f"Refusing to replace modified file: {name}")
        for name in sorted(FILES):
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes((stage / name).read_bytes())
        verify(ROOT)
    print("Fixed data verified and ready in data/.")


if __name__ == "__main__":
    main()
