"""
Download CIHI (Canadian Institute for Health Information) Data Files
=====================================================================
Downloads two real Canadian health data Excel files from CIHI:

  1. CIHI Indicator Library – All Indicator Data
     URL: https://www.cihi.ca/sites/default/files/document/
          indicator-library-all-indicator-data-en.xlsx

  2. 827 – All Patients Readmitted to Hospital Data Table
     URL: https://www.cihi.ca/sites/default/files/document/data-file/
          827-all-patients-readmitted-to-hospital-data-table-en.xlsx

Usage:
    python data/download_cihi_data.py

Outputs:
    data/indicator-library-all-indicator-data-en.xlsx
    data/827-all-patients-readmitted-to-hospital-data-table-en.xlsx

Data Source:
    Canadian Institute for Health Information (CIHI)
    https://www.cihi.ca/en/indicators/all-patients-readmitted-to-hospital
"""

import os
import sys

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CIHI_FILES = [
    {
        "name": "CIHI Indicator Library – All Indicator Data",
        "url": (
            "https://www.cihi.ca/sites/default/files/document/"
            "indicator-library-all-indicator-data-en.xlsx"
        ),
        "filename": "indicator-library-all-indicator-data-en.xlsx",
    },
    {
        "name": "827 – All Patients Readmitted to Hospital Data Table",
        "url": (
            "https://www.cihi.ca/sites/default/files/document/data-file/"
            "827-all-patients-readmitted-to-hospital-data-table-en.xlsx"
        ),
        "filename": "827-all-patients-readmitted-to-hospital-data-table-en.xlsx",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CIHI-Data-Downloader/1.0; "
        "+https://github.com/satyam-kalra/capstone-project)"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def download_file(url: str, dest_path: str, name: str) -> bool:
    """Download a file from *url* to *dest_path*.

    Returns True on success, False on failure.
    """
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Already exists ({size_mb:.1f} MB) – skipping download.")
        return True

    print(f"  Downloading from:\n    {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        print(f"  ERROR: Could not connect to CIHI servers.\n  Details: {exc}")
        return False
    except requests.exceptions.Timeout:
        print("  ERROR: Request timed out after 120 s.")
        return False
    except requests.exceptions.HTTPError as exc:
        print(f"  ERROR: HTTP {exc.response.status_code} – {exc}")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"  ERROR: Unexpected download error – {exc}")
        return False

    # Stream-write to disk
    total = 0
    with open(dest_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"  Saved → {dest_path}  ({size_mb:.1f} MB)")
    return True


def inspect_excel(path: str) -> None:
    """Print basic metadata about an Excel workbook."""
    try:
        import pandas as pd
        from openpyxl.utils.exceptions import InvalidFileException

        xl = pd.ExcelFile(path, engine="openpyxl")
        print(f"  Sheets ({len(xl.sheet_names)}): {xl.sheet_names}")
        for sheet in xl.sheet_names[:3]:  # preview first 3 sheets
            try:
                df = pd.read_excel(
                    path, sheet_name=sheet, nrows=5, engine="openpyxl"
                )
                print(
                    f"\n  Sheet '{sheet}' – {df.shape[1]} columns, "
                    f"preview (first 5 rows):"
                )
                print(df.to_string(max_cols=6, max_colwidth=30))
            except (ValueError, pd.errors.EmptyDataError) as exc:
                print(f"  Could not preview sheet '{sheet}': {exc}")
    except (FileNotFoundError, InvalidFileException, ValueError) as exc:
        print(f"  Could not inspect file: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print("CIHI Data Downloader")
    print("Canadian Institute for Health Information (CIHI)")
    print("=" * 70)

    all_ok = True
    downloaded = []

    for file_info in CIHI_FILES:
        print(f"\n[File] {file_info['name']}")
        dest = os.path.join(DATA_DIR, file_info["filename"])
        ok = download_file(file_info["url"], dest, file_info["name"])
        all_ok = all_ok and ok
        if ok:
            downloaded.append(dest)
            print("\n  Inspecting workbook …")
            inspect_excel(dest)

    print("\n" + "=" * 70)
    if all_ok:
        print(f"Download complete.  {len(downloaded)} file(s) ready in {DATA_DIR}/")
        print("\nNext step:")
        print("  python notebooks/capstone_analysis.py")
    else:
        print(
            "One or more downloads failed.  Check your internet connection "
            "and try again."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
