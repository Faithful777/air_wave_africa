"""Gemma text-generation pipeline and broadcast orchestration."""

import time
import torch

from src.prompts    import build_segment_prompt, build_music_prompt
from src.formatters import format_speech, format_music


# Module-level handles; populated by init_gemma()
_gemma_processor = None
_gemma_model     = None


def init_gemma(processor, model):
    global _gemma_processor, _gemma_model
    _gemma_processor = processor
    _gemma_model     = model


def generate(messages, max_new_tokens=512):
    """Call Gemma 4 and return the plain response string."""
    text = _gemma_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs    = _gemma_processor(text=text, return_tensors="pt").to(_gemma_model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = _gemma_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,
        )
    raw    = _gemma_processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = _gemma_processor.parse_response(raw)
    return parsed.get("text", raw).strip()


def generate_broadcast_content(CONFIG, retriever):
    domain = CONFIG["domain"]
    region = CONFIG["region"]
    mode   = CONFIG["mode"]

    query = (f"{domain} information for rural communities in {region}. "
             f"Practical guidance, simple language, radio broadcast.")
    print("🔍 Retrieving relevant chunks...")
    retrieved = retriever.retrieve(query, domain, top_k=CONFIG["top_k"])
    print()

    # ---- Speech mode: one segment per chunk ----
    if mode == "speech":
        words_per_seg = CONFIG["total_words"] // len(retrieved)
        remainder     = CONFIG["total_words"] - (words_per_seg * len(retrieved))

        segments = []
        for i, chunk in enumerate(retrieved):
            seg_num   = i + 1
            seg_words = words_per_seg + (remainder if seg_num == len(retrieved) else 0)

            print(f"📝 Generating segment {seg_num}/{len(retrieved)} "
                  f"[{chunk.chunk_id}] — {seg_words} words...")

            messages = build_segment_prompt(
                chunk        = chunk,
                domain       = domain,
                region       = region,
                words        = seg_words,
                position     = seg_num,
                total_chunks = len(retrieved),
            )

            t_start = time.time()
            text    = generate(messages, max_new_tokens=seg_words * 3)
            elapsed = time.time() - t_start
            wc      = len(text.split())
            print(f"   ✅ Done — {wc} words in {elapsed:.1f}s")

            segments.append({
                "segment_number": seg_num,
                "chunk_id":       chunk.chunk_id,
                "chunk_heading":  chunk.heading,
                "target_words":   seg_words,
                "word_count":     wc,
                "gen_time_secs":  round(elapsed, 1),
                "text":           text,
            })

        RESULT = format_speech(segments, CONFIG)
        print(f"\n✅ All {len(segments)} segments generated.")
        print(f"   Total words : {RESULT['actual_words']} (target: {RESULT['target_words']})")
        print(f"   Est. TTS    : ~{RESULT['est_tts_secs']}s")
        print(f"   Gen time    : {RESULT['total_gen_secs']}s")
        return RESULT

    # ---- Music mode: single pass ----
    print("🎵 Building music prompt...")
    messages = build_music_prompt(CONFIG, retrieved)
    print("🎤 Generating song with Gemma 4...")
    t0  = time.time()
    raw = generate(messages, max_new_tokens=768)
    print(f"   ✅ Done in {time.time() - t0:.1f}s")
    return format_music(raw, CONFIG, retrieved)
