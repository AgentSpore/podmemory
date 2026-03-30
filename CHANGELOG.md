# Changelog

All notable changes to PodMemory are documented here.

## v0.4.0 (2026-03-30)

### Added
- **Article URL support**: trafilatura extraction from any website (Habr, Medium, Wikipedia, blogs)
- **PDF upload**: PyMuPDF text extraction with page structure
- **EPUB upload**: ebooklib + BeautifulSoup, chapter-aware extraction
- **Chapter picker UI**: select chapters individually, analyze each, merge results
- **Vocabulary extraction**: key terms + definitions for articles/books
- **Quotes extraction**: notable quotes from text content
- **`POST /api/extract-pdf`**: PDF text extraction endpoint
- **`POST /api/extract-epub`**: EPUB extraction with chapters
- **`POST /api/analyze-chapter`**: per-chapter book analysis
- **Separate LLM prompt** for text content (vocabulary + quotes, no timestamps)
- **`source_type` field**: video/article/pdf/book — auto-detected
- **`VocabTerm` schema**: Pydantic model for vocabulary terms
- E2E Playwright tests (homepage, article, YouTube, EPUB, Anki export)

### Changed
- Hero text: "Remember everything you **learn**" (was "watch")
- URL placeholder: "YouTube, article, Habr, VK, Rutube, Wikipedia..."
- Drop zone accepts PDF + EPUB alongside audio/video
- `AnalysisResponse` extended with `source_type`, `vocabulary`, `quotes`

### Dependencies
- Added: `trafilatura>=2.0`, `pymupdf>=1.25`, `ebooklib>=0.18`, `beautifulsoup4>=4.12`

---

## v0.3.0 (2026-03-30)

### Added
- **14 video platforms**: VK Video, Rutube, Dzen, OK.ru, Vimeo, SoundCloud, Twitch, Rumble, Telegram, Apple Podcasts, Bilibili, Coub, Yandex Music
- **SSE progress bar** (`POST /api/analyze-stream`): real-time stages via Server-Sent Events
- **Groq Whisper cascade**: `whisper-large-v3-turbo` → `whisper-large-v3` on 403/429/503
- **Browser Whisper fallback**: `/api/temp-audio/{id}` serves audio when Groq unavailable
- **Subtitle fast-path**: yt-dlp metadata extraction without download (~2s for Rutube)
- **Transcript cleanup**: removes `[Music]`, duplicates, noise from auto-generated subtitles
- **Smart truncation**: 70% head + 30% tail at sentence boundaries
- **LRU cache**: 50 entries, 1h TTL — repeat requests instant
- **Model fallback**: primary LLM → 3 free fallback models on failure
- **Anki export** (`POST /api/export/anki`): .apkg via genanki
- **loguru** replaces stdlib logging across all modules
- **44 integration tests** (63 total): SSE, Anki, platform detection, video ID validation

### Security
- `tempfile.mkstemp()` instead of `mktemp()` (race condition fix)
- Video ID regex validation (path traversal prevention)
- Generic error messages (no internal detail leakage)
- `AnkiExportRequest` Pydantic model instead of raw dict
- `asyncio.Lock` on YouTube rate limiter
- `GROQ_API_KEY` required at startup (Pydantic validator)
- Upload size pre-check via `file.size` before reading into memory
- RFC 5987 Content-Disposition for non-ASCII filenames (Anki + Cyrillic)

### Changed
- `pytubefix` → `yt-dlp` (faster, more reliable)
- `whisper-large-v3` → `whisper-large-v3-turbo` (1.5x faster, same quality)
- yt-dlp audio extraction: ffmpeg postprocessor (64kbps mp3)
- `download_ranges` limit: first 15 min of audio

### Dependencies
- Added: `yt-dlp>=2024.0`, `genanki>=0.13`, `loguru>=0.7`, `sse-starlette>=2.0`
- Removed: `pytubefix`

---

## v0.1.1 (2026-03-27)

### Added
- yt-dlp replaces pytubefix for YouTube audio download
- Groq Whisper cascade: server-side audio transcription
- YouTube rate limiting (5s interval, datacenter IP protection)
- ffmpeg in Dockerfile for yt-dlp audio extraction
- Smoke tests (pytest + TestClient)
- DEEP.md architecture documentation

---

## v0.1.0 (2026-03-26)

### Added
- Initial release
- YouTube transcript extraction via `youtube-transcript-api`
- LLM analysis via OpenRouter (27+ free models)
- Flashcard generation with SM-2 spaced repetition (client-side)
- Neobrutalist UI: Clash Display, Plus Jakarta Sans, IBM Plex Mono
- Light (warm paper) + Dark (chocolate) themes
- Browser Whisper.js fallback for audio transcription
- Library with search, export/import JSON
- Download analysis as .md
- Keyboard shortcuts (Enter, Space, 1-4)
- PWA manifest
- i18n: EN, RU, ZH
- Stateless server (no database, localStorage persistence)
