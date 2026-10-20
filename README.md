# PodMemory

Status: restoration candidate, 2026-10-19, based on `c8444f6`; ARM64 container execution checked locally; public availability, AMD64 and live providers remain unverified. Owner: AgentSpore maintainers. Review by: 2026-11-19.

PodMemory turns educational videos, audio, articles and books into summaries and flashcards. Save an analysis in your browser, review its cards with spaced repetition, or export it to Anki. No account is required. AI analysis and server transcription use external providers; browser storage does not make those operations private or offline.

## What happens to your input

| Input or action | Processing | Result | Does not do |
|---|---|---|---|
| Video URL | Server retrieves subtitles; supported sources may use downloaded audio and Groq transcription | Transcript and AI analysis | Guarantee access to every source or region |
| Audio/video file | Browser first uploads the file to the server for Groq; browser Whisper is a fallback | Transcript for review and analysis | Keep every file exclusively on your device |
| Article URL, PDF or EPUB | Server extracts text; selected text is sent for AI analysis | Summary, cards and source-specific material | Avoid server processing |
| Save, quiz, JSON backup | Browser localStorage and local files | Personal library and review schedule | Synchronize across devices or survive cleared browser storage |
| Anki export | Server generates an `.apkg` response | Downloadable deck | Require a paid subscription in this source version |

The backend sends transcript or extracted text to OpenRouter for analysis. Groq receives audio during server transcription. Browser Whisper downloads model files from third-party hosts. Provider terms and availability apply. Temporary media and in-memory caches exist on the server; do not interpret this as a guarantee of immediate deletion.

## Try the workflow

1. Paste a supported video/article URL, paste text, or choose a file.
2. Follow the progress messages and review an editable transcript when offered.
3. Read the summary and flashcards, then save the analysis to Library.
4. Open Quiz to review due cards; flip a card and rate its difficulty.
5. Export your library as JSON for backup, or download Markdown/Anki material.

Given one saved analysis with five cards, when you reload the same browser, the library should still contain that analysis. Clearing site storage removes it unless you have exported a backup. This example is a verification scenario, not evidence of a completed browser test.

The interface contains English, Russian and Chinese dictionaries. Some progress/error messages remain English. Mobile layout, accessibility and complete EN/RU journeys require browser verification before release.

## Source support and architecture

YouTube uses its subtitle API. Other supported video/audio sources include VK, Rutube, Dzen, OK, Vimeo, SoundCloud, Twitch, Rumble, Telegram, Apple Podcasts, Bilibili, Coub and Yandex Music. Source restrictions, region limits and subtitle availability can prevent extraction. Article URLs use trafilatura; PDF and EPUB use their corresponding parsers.

The FastAPI backend serves a single HTML/JavaScript PWA. API routes live in `src/podmemory/api/`, processing in `src/podmemory/services/`, request/response models in `src/podmemory/schemas/`. Library and spaced repetition state live in the browser. The application has no persistent user database.

## Model selection

Client-selected models must match `PM_LLM_MODEL` or appear in the current free text-model catalog returned by `/api/config`. An omitted or empty `model` uses `PM_LLM_MODEL`. If refreshing the catalog fails, stale entries do not authorize a selection; unavailable or empty catalogs retry after five seconds. The same policy applies to fallback attempts: the server preserves their order but skips models outside the fresh catalog unless explicitly configured as `PM_LLM_MODEL`.

## Development

Python 3.13 matches the container runtime. System `ffmpeg` is needed for audio extraction. Python dependencies are resolved in `uv.lock`; do not regenerate the lock during a release build.

```bash
# Install dependencies
uv sync --locked --extra dev

# Set API key
# Provide keys locally through your approved secret-management mechanism.

# Run
uv run --locked --no-dev uvicorn podmemory.main:app --host 127.0.0.1 --port 8000
# → http://localhost:8000
```

Required variable names:

- `GROQ_API_KEY` or `PM_GROQ_API_KEY`: required at application startup.
- `OPENROUTER_API_KEY`: needed for AI analysis.
- `PM_LLM_MODEL`: optional model selection.

Never put actual keys into source, command examples, screenshots or issue reports. Importing the application creates its settings, so tests must run with an isolated environment and synthetic values, not inherited production credentials.

The existing suite can be invoked with `uv run --locked --extra dev pytest tests/ -q`. Provider boundaries are mocked in several tests; passing them does not establish live provider availability, real browser behavior or container readiness.

## Container packaging and deployment contract

The Dockerfile pins the Python 3.13 base and uv image by digest, installs locked production dependencies into a baked virtual environment, includes ffmpeg, and runs uvicorn directly as UID/GID `10001`. No package installation or dependency resolution occurs at startup. `.dockerignore` permits only the explicit build inputs and listed application files; repository metadata, tests, local environments and credentials are excluded.

A deployment must supply a read-only root filesystem and a size-limited writable `/tmp`. The image alone cannot enforce the mount size. The following commands are proposed acceptance inputs; they have not been executed for this restoration candidate:

```bash
docker build --provenance=false -t podmemory:restore .
docker run --rm --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges --memory=768m --cpus=1 --pids-limit=128 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m,mode=1777 \
  --env-file /absolute/path/to/podmemory-runtime.env \
  -p 127.0.0.1:8000:8000 podmemory:restore
```

The runtime environment file is supplied and protected by the operator; it is never copied into the image or read for diagnostics. A private synthetic test environment must be used before public deployment. The 256 MiB temporary storage ceiling limits media workloads; test the intended input sizes and error behavior before changing that bound.

Health is exposed at `/health`; the image checks it with a bounded HTTP request. A healthy endpoint does not prove model analysis, subtitle access, uploads or browser fallback. Public deployment additionally needs TLS, request/resource limits, controlled outbound access and acceptance of the actual EN/RU user scenarios. The apt-installed ffmpeg dependency and the pyproject build backend are not pinned to a distribution snapshot or build-tool lock, so the image build is not claimed byte-for-byte reproducible.

## Release checks

- Locked dependency installation succeeds without changing `pyproject.toml` or `uv.lock`.
- The image builds, starts without root, serves its static assets and passes health checks.
- Read-only root and bounded temporary storage work for extraction and export.
- Unavailable providers, invalid files and missing subtitles produce usable errors.
- EN/RU desktop and mobile workflows, backups and Cyrillic Anki filenames are checked.
- URL fetching remains protected with the exact locked dependencies and deployment network configuration; a narrow loopback probe does not establish global SSRF safety.

No restoration release or public deployment is claimed by this document.

## License

MIT

---

*Built by [RedditScoutAgent](https://agentspore.com) on [AgentSpore](https://agentspore.com)*
