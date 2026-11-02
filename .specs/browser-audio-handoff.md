# Browser audio handoff

## Status
Local candidate in development; coordinator review required before publication.

## Problem
The existing non-YouTube SSE transcription fallback advertises a temporary audio URL, then deletes the file in the generator cleanup. The existing browser consumer cannot download it. This is distinct from EPUB extraction, backup import and the open yt-dlp proxy work.

## Acceptance
- Preserve downloaded audio across supported server-transcription fallback until one HTTP download or five-minute expiry.
- Register ownership before publishing the URL; preserve exact bytes and existing audio/mpeg response contract.
- One successful claim only; unknown, consumed, missing or expired IDs return404.
- Delete owned audio and its empty temporary directory after normal response, send failure, cancellation or range response; expiry must not interrupt an active download.
- Retain cleanup on ordinary server transcription success, ordinary errors, oversized input, unreadable files and generator cancellation after download.
- Keep API payloads, browser code, provider/model policy and dependencies unchanged.

## Validation
Real SSE and file HTTP routes with synthetic local audio and mocked download/transcription/analysis boundaries; external sockets forbidden. RED before fix, targeted lifecycle/error probes, full existing Python suite, exact-diff independent review. No browser/model/provider/production calls claimed. Cancellation during the downloader thread itself is outside this patch.

## Aggregate ownership correction
Coordinator rejected the first candidate: pending and active handoffs had no aggregate count/bytes bound. Preserve that worktree and its144-test receipt as historical.

Use a local budget of at most8 retained accepted files and50MiB of accepted audio, counting pending AND claimed/streaming ownership until file removal. Admission is synchronous; no accepted offer is evicted. Refuse a new handoff with a clear busy SSE error and clean up its just-downloaded file. Preserve five-minute pending expiry, one-shot HTTP download and active-response protection. This limits retained handoffs, not downloader concurrency or global filesystem usage.

Tests use tiny files and low overridden budgets: count/bytes/exact boundary, active-send plus pending pressure, expiry, refusal cleanup/capacity reuse, concurrent admission, missing/read/send/range/disconnect/cancellation, and idempotent release. Existing EPUB and prior accepted features remain untouched.

## Ownership limits and cleanup failures
Reservation totals are computed from the small ledger, not separately mutable counters. Successful removal or an already missing file releases ownership exactly once; failure to remove an empty parent directory does not keep audio bytes charged. Failed deletion of an accepted file keeps its reservation and retries locally.

Refused audio is removed before yielding the busy event, so holding that SSE send cannot retain a normally removable refused file. As with the pre-existing downloaded-file cleanup, deletion of an unreserved/refused file remains best effort on filesystem failure. This is not a hard global disk cap or a downloader-concurrency limit; process termination, cross-worker handoff and downloader-thread cancellation before returning a path remain outside this patch.
