"""Download helper for denoising benchmark datasets.

Supports:
* **BSDS500** — Berkeley Segmentation Dataset (400 train + 200 test).
* **Kodak24** — 24 classic photographic test images.

Usage::

    python -m data.download          # download all
    python -c "from data.download import setup_datasets; setup_datasets()"
"""

import os
import tarfile
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import requests; provide fallback guidance if missing
try:
    import requests
except ImportError:
    requests = None  # type: ignore


def _ensure_requests():
    if requests is None:
        raise ImportError(
            "The 'requests' library is required for downloading datasets.\n"
            "Install it with: pip install requests"
        )


# ---------------------------------------------------------------------------
# BSDS500
# ---------------------------------------------------------------------------
BSDS500_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
    "BSR/BSR_bsds500.tgz"
)

BSDS500_MIRROR = (
    "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
)


def download_bsds500(data_root: str = "./datasets") -> str:
    """Download and extract BSDS500 images.

    Creates::

        data_root/BSDS500/train/   (200 images)
        data_root/BSDS500/test/    (200 images)

    Args:
        data_root: Root directory for datasets.

    Returns:
        Path to the BSDS500 directory.
    """
    _ensure_requests()
    dest = os.path.join(data_root, "BSDS500")
    train_dir = os.path.join(dest, "train")
    test_dir = os.path.join(dest, "test")

    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) >= 100:
        logger.info("BSDS500 already downloaded at %s", dest)
        return dest

    os.makedirs(dest, exist_ok=True)
    tgz_path = os.path.join(data_root, "BSR_bsds500.tgz")

    # Download
    logger.info("Downloading BSDS500 from %s ...", BSDS500_URL)
    try:
        resp = requests.get(BSDS500_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(tgz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        logger.info("Download complete (%s)", tgz_path)
    except Exception as exc:
        logger.warning("Download failed: %s", exc)
        _print_manual_instructions_bsds(dest)
        return dest

    # Extract
    logger.info("Extracting ...")
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_root)
    except Exception as exc:
        logger.warning("Extraction failed: %s", exc)
        _print_manual_instructions_bsds(dest)
        return dest

    # Organise: BSR/BSDS500/data/images/{train,test,val}/ → dest/{train,test}/
    img_base = os.path.join(data_root, "BSR", "BSDS500", "data", "images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for split_src, split_dst in [("train", train_dir), ("val", train_dir), ("test", test_dir)]:
        src_dir = os.path.join(img_base, split_src)
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(split_dst, fname)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

    # Cleanup
    bsr_dir = os.path.join(data_root, "BSR")
    if os.path.isdir(bsr_dir):
        shutil.rmtree(bsr_dir, ignore_errors=True)
    if os.path.isfile(tgz_path):
        os.remove(tgz_path)

    n_train = len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0
    n_test = len(os.listdir(test_dir)) if os.path.isdir(test_dir) else 0
    logger.info("BSDS500 ready: %d train, %d test images at %s", n_train, n_test, dest)
    return dest


def _print_manual_instructions_bsds(dest: str):
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  BSDS500 automatic download failed.                        ║\n"
        "║                                                            ║\n"
        "║  Please download manually:                                 ║\n"
        f"║  1. Go to: {BSDS500_URL}  ║\n"
        "║  2. Extract the archive.                                   ║\n"
        f"║  3. Place images in:                                       ║\n"
        f"║     {dest}/train/   (training images)               ║\n"
        f"║     {dest}/test/    (test images)                   ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )


# ---------------------------------------------------------------------------
# Kodak24
# ---------------------------------------------------------------------------
KODAK_BASE_URL = "http://r0k.us/graphics/kodak/kodak/"


def download_kodak24(data_root: str = "./datasets") -> str:
    """Download 24 Kodak test images.

    Creates ``data_root/Kodak24/`` with ``kodim01.png`` … ``kodim24.png``.

    Args:
        data_root: Root directory for datasets.

    Returns:
        Path to the Kodak24 directory.
    """
    _ensure_requests()
    dest = os.path.join(data_root, "Kodak24")

    if os.path.isdir(dest) and len(os.listdir(dest)) >= 24:
        logger.info("Kodak24 already downloaded at %s", dest)
        return dest

    os.makedirs(dest, exist_ok=True)

    for i in range(1, 25):
        fname = f"kodim{i:02d}.png"
        out_path = os.path.join(dest, fname)
        if os.path.isfile(out_path):
            continue
        url = KODAK_BASE_URL + fname
        logger.info("  Downloading %s ...", fname)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", fname, exc)

    n = len([f for f in os.listdir(dest) if f.endswith(".png")])
    logger.info("Kodak24 ready: %d images at %s", n, dest)
    return dest


# ---------------------------------------------------------------------------
# Combined setup
# ---------------------------------------------------------------------------
def setup_datasets(data_root: str = "./datasets") -> dict[str, str]:
    """Download all datasets.

    Returns:
        Dict mapping dataset name to its directory path.
    """
    paths = {}
    paths["bsds500"] = download_bsds500(data_root)
    # paths["kodak24"] = download_kodak24(data_root)
    return paths


if __name__ == "__main__":
    setup_datasets()
