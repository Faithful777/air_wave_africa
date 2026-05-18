"""Format raw model output into structured broadcast dicts."""

import re

from src.config import WORDS_PER_SECOND


def format_speech(segments, cfg):
    full_text  = "\n\n".join(s["text"] for s in segments)
    total_wc   = sum(s["word_count"]   for s in segments)
    total_time = sum(s["gen_time_secs"] for s in segments)
    return {
        "mode":          "speech_broadcast",
        "domain":        cfg["domain"],
        "region":        cfg["region"],
        "target_words":  cfg["total_words"],
        "actual_words":  total_wc,
        "est_tts_secs":  round(total_wc / WORDS_PER_SECOND),
        "total_gen_secs": round(total_time, 1),
        "segments":       segments,
        "broadcast_text": full_text,
    }


def format_music(raw, cfg, chunks):
    sections = {}
    labels   = ["MUSIC_DESCRIPTION", "CHORUS", "VERSE_1", "VERSE_2", "VERSE_3"]
    pattern  = r"(" + "|".join(labels) + r")\s*:\s*\n"
    parts    = re.split(pattern, raw)
    i = 1
    while i < len(parts) - 1:
        sections[parts[i].strip()] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2
    return {
        "mode":              "music_broadcast",
        "domain":            cfg["domain"],
        "region":            cfg["region"],
        "duration_secs":     cfg["duration_secs"],
        "sources":           [c.chunk_id for c in chunks],
        "music_description": sections.get("MUSIC_DESCRIPTION", "").strip(),
        "chorus":            sections.get("CHORUS", "").strip(),
        "verse_1":           sections.get("VERSE_1", "").strip(),
        "verse_2":           sections.get("VERSE_2", "").strip(),
        "verse_3":           sections.get("VERSE_3", "").strip(),
        "full_raw":          raw,
    }
