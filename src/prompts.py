"""System and per-task prompt builders."""

SYSTEM_PROMPT = """\
You are a broadcast writer for rural communities in sub-Saharan Africa.
Your job is to take accurate information and turn it into broadcasts that:
- Use very simple English that a person with no education can understand
- Avoid long sentences — use short ones
- Use everyday words — never technical words
- Speak directly to the listener as 'you' and 'we'
- Feel warm, friendly, and trustworthy
- Include only true, helpful information from the reference material
- Output ONLY the requested content and nothing else"""


def build_segment_prompt(chunk, domain, region, words, position, total_chunks):
    """Build a prompt for a SINGLE chunk → SINGLE speech segment."""
    if position == 1:
        position_note = (
            "This is the FIRST segment of the broadcast. "
            "Open with a warm, engaging greeting to listeners."
        )
    elif position == total_chunks:
        position_note = (
            "This is the LAST segment of the broadcast. "
            "Close warmly — encourage listeners and thank them for tuning in."
        )
    else:
        position_note = (
            f"This is segment {position} of {total_chunks}. "
            "Continue naturally from the previous topic. "
            "Do not open with a greeting — just move into this topic smoothly."
        )

    user_msg = f"""\
REFERENCE INFORMATION FOR THIS SEGMENT:
[{chunk.chunk_id}] {chunk.heading}
{chunk.body[:900]}

YOUR TASK:
Write ONE segment of a {domain.upper()} radio broadcast for rural communities in: {region}.

Segment position note: {position_note}

Rules:
- Exactly {words} words (count carefully)
- Very simple English — short sentences — everyday words only
- Speak directly to the listener as 'you' and 'we'
- Sound like a friendly, caring radio presenter
- Use only facts from the reference information above
- Do NOT include any title, label, or heading
- Do NOT start with 'Segment' or any numbering
- Do NOT add explanations or notes after the segment ends

Write this segment now:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def build_music_prompt(cfg, chunks):
    from src.config import LYRIC_DENSITY
    context = "\n\n---\n\n".join(
        f"[{c.chunk_id}] {c.heading}\n{c.body[:600]}" for c in chunks
    )
    target_words = int(cfg["duration_secs"] * LYRIC_DENSITY)
    user_msg = f"""\
REFERENCE INFORMATION:
{context}

YOUR TASK:
Create an educational song in English about {cfg['domain'].upper()} for rural communities in: {cfg['region']}.
The song will be {cfg['duration_secs']} seconds long.

OUTPUT FORMAT — output exactly these four sections with these exact labels:

MUSIC_DESCRIPTION:
(2-3 sentences: musical style, tempo, instruments, mood.
 Use common local musical styles for the region. Mostly fast paced drum-heavy Afrobeats)

CHORUS:
(2-4 lines — catchy, repeating, states the single most important message.
 Very simple words. Easy to remember and sing.)

VERSE_1:
(~{target_words // 3} words — the problem or situation the listener faces.)

VERSE_2:
(~{target_words // 3} words — the solution or action to take.)

VERSE_3:
(~{target_words // 3} words — the solution or action to take.)

Lyrics rules: very simple English, short singable lines, speak as 'you' and 'we', rhyme where natural, facts only.

Write the song now:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
