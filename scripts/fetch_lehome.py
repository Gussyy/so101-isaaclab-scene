# SPDX-License-Identifier: BSD-3-Clause
"""Download the LeHome household asset library into ``assets/lehome/``.

LeHome (https://github.com/lehome-official/lehome) publishes its simulation assets as a public
HuggingFace dataset. The assets are plain USD, so they load here even though LeHome's *code* does
not -- it targets Isaac Lab 2.3 with PhysX particle systems, and this repo is Isaac Lab 3.0 on
Newton. See docs/LEHOME.md for what did and did not come across.

The full dataset is ~6.6 GB and most of that is a one-bedroom apartment this arm cannot reach
across. The default here is ``objects`` + ``Material`` (~1.7 GB); the apartment is opt-in.

Usage:
    python scripts/fetch_lehome.py                  # objects + materials, ~1.7 GB
    python scripts/fetch_lehome.py --what scene     # the 4.8 GB apartment as well
    python scripts/fetch_lehome.py --list           # what is on the server, with sizes
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "lehome/lehome_release"
DEST = Path(__file__).resolve().parent.parent / "assets" / "lehome"

# What each name pulls. Kept explicit so `--what everything` is a decision, not a default.
GROUPS = {
    "objects": ["objects/**"],       # ~1.5 GB: the prop library
    "material": ["Material/**"],     # ~160 MB: MDL materials the props bind to
    "scene": ["scenes/**"],          # ~4.8 GB: the 1-bedroom apartment
    "robots": ["robots/**"],         # ~94 MB: LeHome's own SO-101 USD. We have our own.
}
DEFAULT = ["objects", "material"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="*", default=DEFAULT, choices=sorted(GROUPS), help=f"default: {' '.join(DEFAULT)}")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--list", action="store_true", help="print sizes and exit, download nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    if args.list:
        api = HfApi()
        totals: dict[str, int] = {}
        for f in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
            if getattr(f, "size", None) is None:
                continue
            parts = f.path.split("/")
            for depth in (1, 2):
                totals["/".join(parts[:depth])] = totals.get("/".join(parts[:depth]), 0) + f.size
        for key in sorted(totals):
            print(f"{totals[key] / 1e6:10.1f} MB  {key}")
        return

    patterns = [p for name in args.what for p in GROUPS[name]]
    # `.thumbs` is Omniverse's browser-thumbnail cache: hundreds of PNGs the simulator never
    # opens. Excluding it is not cosmetic, it is a meaningful slice of the download.
    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(args.dest),
        allow_patterns=patterns,
        ignore_patterns=["**/.thumbs/**"],
    )
    print(f"\nassets in {path}")
    print("next:  python scripts/measure_lehome.py      # bounding boxes -> the catalogue")


if __name__ == "__main__":
    main()
