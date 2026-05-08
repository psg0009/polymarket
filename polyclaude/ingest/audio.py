"""Audio transcription via faster-whisper.

Heavy dep — lazy-loaded. Falls back to a no-op transcriber if not installed,
which keeps the rest of the pipeline running on minimal installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from polyclaude.ingest.normalizer import RawEvent
from polyclaude.ledger.db import Modality
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _model(model_size: str = "small.en") -> Any | None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.info("ingest.whisper.unavailable")
        return None
    try:
        return WhisperModel(model_size, compute_type="int8")
    except Exception as e:  # pragma: no cover
        log.warning("ingest.whisper.load_failed", err=str(e))
        return None


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    confidence: float


def transcribe(audio_path: str, model_size: str = "small.en") -> Transcript | None:
    m = _model(model_size)
    if m is None:
        return None
    segments, info = m.transcribe(audio_path, beam_size=5)
    parts = []
    confs = []
    for seg in segments:
        parts.append(seg.text)
        if getattr(seg, "avg_logprob", None) is not None:
            confs.append(seg.avg_logprob)
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return Transcript(
        text=" ".join(parts).strip(),
        language=info.language,
        duration_s=float(info.duration),
        confidence=avg_conf,
    )


def transcript_to_event(
    audio_url: str, source: str, transcript: Transcript, ts: datetime | None = None,
    is_india: bool = False,
) -> RawEvent:
    return RawEvent(
        source=source,
        text=transcript.text,
        title=None,
        url=audio_url,
        ts=ts or datetime.now(timezone.utc),
        audio_url=audio_url,
        transcript_meta={
            "language": transcript.language,
            "duration_s": transcript.duration_s,
            "avg_logprob": transcript.confidence,
        },
        modality=Modality.audio,
        lang=transcript.language,
        is_india=is_india,
    )
