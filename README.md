# Soybrary

Web scraper and gallery server for building your own local Soyjak library.

![Preview](preview%20soybrary.png)

## Setup

```bash
./setup.sh     # venv, dependencies, Playwright Chromium
./start.sh     # serves http://127.0.0.1:8000
```

Scrape from the gallery's **Scrape** button, or standalone:

```bash
python scraper.py --start 1000 --limit 500
```

Both share one library, so a run started in the UI resumes where the CLI left
off. `ffmpeg`/`ffprobe` are optional — without them videos are validated by
signature only and get no thumbnails.

Settings live in `config.json`: `concurrency`, `delay_ms`, `data_dir`,
`validate_images`/`validate_videos`, `sanitize_images`/`sanitize_videos`,
`pregenerate_thumbnails`, `thumbnail_size`, `host`/`port`. The scrape endpoints
are unauthenticated, so only widen `host` on a network you trust.

## Library

`data/` holds `images/`, `videos/`, `thumbnails/`, `metadata/` and
`soybooru.db` (catalog, FTS5 search index, scrape state). Deleting the database
loses scrape progress; `thumbnails/` regenerates on demand.

Built for a few hundred thousand posts: index-only listing, FTS5 search, WAL so
browsing stays responsive during a scrape, media resolved by post id with
immutable cache headers, and animations fetched only as they approach the
viewport.

Tests: `python -m unittest discover`

## Related

[soy-diffusion](https://github.com/clankerabuse/soy-diffusion) trains an SDXL
LoRA on a Soybrary library; the model is on
[Hugging Face](https://huggingface.co/ChineseWhiteGuy/soy_diffusion).
