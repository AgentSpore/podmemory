# PodMemory -- Changelog

## v0.1.1 (2026-03-27)

- **yt-dlp replaces pytubefix** for YouTube audio download (more reliable, actively maintained)
- **Groq Whisper cascade**: server-side audio transcription via Groq API (whisper-large-v3)
- **YouTube rate limiting**: 5s minimum interval between transcript API requests, datacenter IP protection
- **ffmpeg added to Docker** for yt-dlp audio extraction
- **Smoke tests** added (pytest + TestClient): health, config, analyze validation, URL parsing, user paste
- **DEEP.md** added: architecture, data flow, SM-2 algorithm, error handling, security model

## v0.1.0 (2026-03-26)

- Initial release
- YouTube transcript extraction via youtube-transcript-api
- LLM analysis via OpenRouter (free models)
- Flashcard generation with spaced repetition (SM-2, client-side)
- Neobrutalist UI (vanilla JS SPA)
- Browser Whisper.js fallback for audio transcription
- Stateless server: no database, localStorage persistence
- Export/import JSON for data portability
