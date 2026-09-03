# Soybrary

A web scraper and image gallery server for creating and managing your own local Soyjak library.

![Preview](preview%20soybrary.png)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. Start the server:

   ```bash
   python server.py
   ```

   Or use `start.sh` / `start.bat`, which also opens the gallery in your browser.

Scraping is driven from the gallery's **Scrape** button, or standalone:

```bash
python scraper.py --start 1000 --limit 500
```

Both entry points share one library, so a run started in the UI resumes where
the CLI left off and vice versa.

`ffmpeg` and `ffprobe` are optional. Without them, videos are validated by
signature only and get no thumbnails.

## Configuration

`config.json` is read by both the scraper and the server:

| Key | Default | Meaning |
| --- | --- | --- |
| `concurrency` | `3` | Posts fetched in parallel |
| `delay_ms` | `2000` | Politeness delay per post, jittered ±30% |
| `data_dir` | `./data` | Library location, relative to the project |
| `validate_images` / `validate_videos` | `true` | Check signatures and structure before storing |
| `sanitize_images` | `true` | Strip metadata by re-encoding stills; animations are left intact |
| `sanitize_videos` | `false` | Remux through ffmpeg to drop container metadata |
| `pregenerate_thumbnails` | `true` | Build thumbnails during the scrape instead of on first view |
| `thumbnail_size` | `300` | Longest thumbnail edge, in pixels |
| `host` / `port` | `127.0.0.1:8000` | Where the server listens. The scrape endpoints are unauthenticated, so only widen this on a network you trust |

## Library layout

```
data/
├── images/      one file per still, named <post id>.<ext>
├── videos/      one file per video
├── thumbnails/  300px JPEG previews
├── metadata/    per-post JSON
└── soybooru.db  catalog, search index and scrape state
```

Deleting `soybooru.db` loses scrape progress; deleting `thumbnails/` is safe,
as they are regenerated on demand.

## How it scales

The catalog is built for a library of a few hundred thousand posts:

- Listing and counting are index-only lookups, and search runs against an FTS5
  index instead of scanning every row. Result totals are cached until the
  library changes.
- The database runs in WAL mode, so browsing stays responsive during a scrape.
- Media files resolve directly from a post id and extension rather than
  scanning a directory that holds one file per post, and they are served with
  an immutable cache header.
- The gallery fetches full-size animations only as they approach the viewport.

## Tests

```bash
python -m unittest discover
```

## Related projects

- **[soy-diffusion](https://github.com/clankerabuse/soy-diffusion)** — SDXL LoRA training pipeline that uses a Soybrary library as its dataset. The trained model is hosted on [Hugging Face](https://huggingface.co/ChineseWhiteGuy/soy_diffusion).
