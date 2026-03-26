# PodMemory -- Architecture Deep Dive

## Data Flow

```
URL or File Upload
        |
        v
+------------------+
| Input Router     |  POST /api/analyze (url|text), POST /api/transcribe (file)
+------------------+
        |
        v
+------------------+
| Transcript Layer |  YouTube subtitles / Groq Whisper / user paste / browser Whisper.js
+------------------+
        |
        v
+------------------+
| LLM Analysis     |  OpenRouter (free models) -> structured JSON
+------------------+
        |
        v
+------------------+
| Response Schema  |  AnalysisResponse (pydantic): title, tldr, flashcards, timestamps...
+------------------+
        |
        v
+------------------+
| Client (browser) |  localStorage persistence, SM-2 spaced repetition, export/import
+------------------+
```

## Transcription Cascade

Three-tier fallback for obtaining transcript text:

1. **YouTube Subtitles** (instant, no key needed)
   - `youtube-transcript-api` fetches existing subtitles
   - Prefers manual transcripts over auto-generated
   - Fails if video has no subtitles at all

2. **Groq Whisper** (server-side, needs GROQ_API_KEY)
   - User uploads audio/video file -> POST /api/transcribe
   - `whisper-large-v3` model, 25MB file limit
   - 10h/month free tier; returns 503 on rate limit

3. **Browser Whisper.js** (client-side, no server)
   - Frontend downloads and runs Whisper WASM locally
   - Last resort fallback, handled entirely in JavaScript
   - No server involvement

yt-dlp audio download (`/api/youtube-audio/{video_id}`) is available but NOT chained into `/api/analyze` automatically -- datacenter IPs get banned by YouTube.

## Rate Limiting Strategy

YouTube aggressively blocks datacenter IPs. Protections:

- **Server-side throttle**: minimum 5s between YouTube transcript requests (`_YT_MIN_INTERVAL`)
- **No auto-download**: yt-dlp endpoint is explicit-only, not triggered from /analyze
- **Graceful degradation**: on subtitle failure, returns `NO_SUBTITLES` error with instructions to upload file instead
- **Client as fallback**: browser-based Whisper.js uses user's residential IP (not blocked)

## SM-2 Algorithm (Client-Side)

Spaced repetition runs entirely in the browser (localStorage). Implementation:

```
On review of flashcard with quality q (0-5):
  if q >= 3: correct
    repetition += 1
    if repetition == 1: interval = 1 day
    elif repetition == 2: interval = 6 days
    else: interval *= easiness_factor
  else: incorrect
    repetition = 0
    interval = 1 day

  easiness_factor += 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
  easiness_factor = max(1.3, easiness_factor)

  next_review = now + interval
```

Cards are stored per-analysis in localStorage with review state.

## Error Handling Flow

```
Input Validation (400)
  -> Missing url AND text: 400
  -> Non-YouTube URL without file: 400

Transcript Errors (400/502)
  -> Video unavailable/restricted: 400
  -> No subtitles found: 400 NO_SUBTITLES
  -> Transcript fetch failure: 400

Groq Whisper Errors (502/503)
  -> No GROQ_API_KEY: 503 GROQ_UNAVAILABLE
  -> Rate limited: 503 GROQ_RATE_LIMIT
  -> Empty file: 400
  -> File too large: 400

LLM Analysis Errors (502)
  -> API call failure: 502
  -> Invalid JSON from LLM: 502
  -> Missing required fields: 502
  -> Timeout: 502
```

## Security Model

- **Stateless server**: no database, no user accounts, no session storage
- **No user data stored**: all analysis results live in browser localStorage
- **Export/import**: JSON file download/upload for backup (client-side only)
- **API keys**: server-side only (OpenRouter, Groq), never exposed to client
- **File uploads**: temp files deleted immediately after transcription
- **No auth required**: public API, no PII collected

## Project Layout

```
src/podmemory/
  main.py              FastAPI app, health check, static mount
  api/
    analyze.py         Endpoints: /config, /analyze, /transcribe, /youtube-audio
  core/
    config.py          Settings via pydantic-settings (PM_ prefix)
  schemas/
    analysis.py        AnalyzeRequest, AnalysisResponse, Flashcard, Timestamp
  services/
    transcript.py      YouTube transcript extraction, URL parsing, user paste
    analyzer.py        LLM analysis via OpenRouter
    audio_transcript.py  Groq Whisper transcription, yt-dlp audio download
  static/
    index.html         Neobrutalist SPA (vanilla JS, Whisper.js, SM-2)
```

## Design Decisions

- **No database**: intentional. Users own their data. Server is a pure processing pipeline.
- **Free models only by default**: config endpoint filters OpenRouter models with `pricing.prompt == "0"`. Users with own API keys can override.
- **Transcript truncation at 15k chars**: free models have limited context windows. Better to analyze 15k well than fail on 100k.
- **yt-dlp over pytubefix**: pytubefix broke constantly due to YouTube cipher changes. yt-dlp is actively maintained and handles more edge cases.
- **ffmpeg in Docker**: required by yt-dlp for audio format conversion (merging streams, extracting audio).
