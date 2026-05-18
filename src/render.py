"""Plain-text (CLI-friendly) rendering of broadcast results."""

def render_speech(r) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(" SPEECH BROADCAST — Per-Chunk Generation")
    lines.append("=" * 60)
    lines.append(f"Domain        : {r['domain'].upper()}")
    lines.append(f"Region        : {r['region']}")
    lines.append(f"Total words   : {r['actual_words']} (target: {r['target_words']})")
    lines.append(f"Est. TTS dur. : ~{r['est_tts_secs']}s "
                 f"(~{r['est_tts_secs']//60}m {r['est_tts_secs']%60}s)")
    lines.append(f"Generation t. : {r['total_gen_secs']}s")
    lines.append(f"Segments      : {len(r['segments'])}")
    lines.append("")
    lines.append("Segment Summary")
    lines.append("-" * 60)
    for s in r["segments"]:
        lines.append(f" #{s['segment_number']} | {s['chunk_id']:10s} | "
                     f"{s['chunk_heading'][:45]:45s} | "
                     f"{s['word_count']:4d} words | {s['gen_time_secs']}s")
    lines.append("")
    lines.append("=" * 60)
    lines.append(" FULL CONCATENATED BROADCAST")
    lines.append("=" * 60)
    lines.append(r["broadcast_text"])
    return "\n".join(lines)


def render_music(r) -> str:
    return (
        "=" * 60 + "\n"
        " MUSIC BROADCAST OUTPUT\n"
        + "=" * 60 + "\n"
        f"Domain      : {r['domain'].upper()}\n"
        f"Region      : {r['region']}\n"
        f"Duration    : {r['duration_secs']}s\n"
        f"KB Sources  : {', '.join(r['sources'])}\n\n"
        f"-- Music Description --\n{r['music_description']}\n\n"
        f"-- Chorus --\n{r['chorus']}\n\n"
        f"-- Verse 1 --\n{r['verse_1']}\n\n"
        f"-- Verse 2 --\n{r['verse_2']}\n\n"
        f"-- Verse 3 --\n{r['verse_3']}\n"
    )
