"""
Data Downloader for Book Recommendation System
Downloads the Goodbooks-10k dataset from GitHub:
- ratings.csv: user_id, book_id, rating
- books.csv: book_id, title, authors, average_rating, ratings_count, image_url, etc.
"""

import os
import sys
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = {
    "ratings.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
    "books.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
}

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:6.2f}% ({mb_downloaded:6.2f} MB / {mb_total:6.2f} MB)")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb_downloaded:6.2f} MB")
    sys.stdout.flush()

def download_file(filename: str, url: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    target_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[ALREADY EXISTS] {filename} ({size_mb:.2f} MB) is already in {DATA_DIR}")
        return True

    print(f"\n[DOWNLOADING] {filename} from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
        print() # Newline after progress
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[SUCCESS] Saved {filename} ({size_mb:.2f} MB) to {target_path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        print()
        print(f"[ERROR] Failed to download {filename}: {e}")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        return False

def main():
    print("=" * 60)
    print("Goodbooks-10k Dataset Downloader")
    print("=" * 60)

    all_ok = True
    for fname, url in FILES.items():
        ok = download_file(fname, url)
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All dataset files downloaded and verified successfully!")
        print(f"Data files are available in: {os.path.abspath(DATA_DIR)}")
    else:
        print("One or more files failed to download automatically.")
        print("Please manually download the missing files and place them in 'data/':")
        for fname, url in FILES.items():
            print(f" - {fname} -> {url}")
    print("=" * 60)

if __name__ == "__main__":
    main()
