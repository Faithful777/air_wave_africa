"""Translation, TTS, and music audio generation."""

import re
import numpy as np
import soundfile as sf
import torch

from src.config import (
    MUSIC_MODEL_ID,
    SENTENCE_PAUSE_SECONDS,
    TRANSLATION_MODEL_NAME,
    tts_model_name,
)


# ============================================================
# SHARED MODEL LOADING (lazy, loaded once)
# ============================================================
_processor          = None
_translation_model  = None
_tts_tokenizer      = None
_tts_model          = None
_target_lang        = None


def init_audio(target_lang: str):
    """Remember which TTS model to load (per language code)."""
    global _target_lang
    _target_lang = target_lang


def _load_translation_models():
    global _processor, _translation_model
    if _processor is None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        _processor = AutoProcessor.from_pretrained(
            TRANSLATION_MODEL_NAME, local_files_only=True
        )
        _translation_model = SeamlessM4Tv2Model.from_pretrained(
            TRANSLATION_MODEL_NAME, device_map="auto", local_files_only=True
        )


def _load_tts_models():
    global _tts_tokenizer, _tts_model
    if _tts_tokenizer is None:
        from transformers import AutoTokenizer, VitsModel
        model_name = tts_model_name(_target_lang)
        _tts_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        _tts_model     = VitsModel.from_pretrained(model_name, local_files_only=True)


# ============================================================
# TEXT UTILITIES
# ============================================================
def _clean_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = text.replace("<turn>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# TRANSLATION
# ============================================================
def _translate_sentence(sentence, src_lang="eng", tgt_lang="yor"):
    _load_translation_models()
    text_inputs = _processor(
        text=sentence,
        src_lang=src_lang,
        return_tensors="pt",
    ).to(_translation_model.device)
    output_tokens = _translation_model.generate(
        **text_inputs,
        tgt_lang=tgt_lang,
        generate_speech=False,
    )
    return _processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)[0]


def _translate_block(text, src_lang="eng", tgt_lang="yor"):
    text      = _clean_text(text)
    sentences = _split_into_sentences(text)
    return " ".join(
        _translate_sentence(s, src_lang=src_lang, tgt_lang=tgt_lang) for s in sentences
    )


# ============================================================
# TTS
# ============================================================
def _generate_speech_waveform(text: str) -> np.ndarray:
    _load_tts_models()
    inputs = _tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = _tts_model(**inputs).waveform
    return waveform.squeeze().cpu().numpy()


def _text_to_audio_tensor(text, src_lang="eng", tgt_lang="yor"):
    _load_tts_models()
    text      = _clean_text(text)
    sentences = _split_into_sentences(text)

    sample_rate = _tts_model.config.sampling_rate
    silence     = np.zeros(int(sample_rate * SENTENCE_PAUSE_SECONDS), dtype=np.float32)

    segments: list[np.ndarray] = []
    for sentence in sentences:
        translated = _translate_sentence(sentence, src_lang=src_lang, tgt_lang=tgt_lang)
        try:
            waveform = _generate_speech_waveform(translated)
            segments.append(waveform)
            segments.append(silence)
        except Exception as exc:
            print(f"   [!] Speech generation failed: {exc}")

    if not segments:
        return torch.zeros(1, 0), sample_rate

    full_audio = np.concatenate(segments)
    max_val    = np.abs(full_audio).max()
    if max_val > 0:
        full_audio /= max_val

    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)
    return audio_tensor, sample_rate


# ============================================================
# LYRICS FORMATTING (music mode)
# ============================================================
def _format_and_translate_lyrics(song_dict, src_lang="eng", tgt_lang="yor"):
    sections: list[str] = []
    chorus_raw = song_dict.get("chorus")
    chorus_translated = (
        _translate_block(chorus_raw, src_lang=src_lang, tgt_lang=tgt_lang)
        if chorus_raw else None
    )

    verse_num = 1
    verses: list[str] = []
    while f"verse_{verse_num}" in song_dict:
        verses.append(song_dict[f"verse_{verse_num}"])
        verse_num += 1

    for i, verse_raw in enumerate(verses):
        verse_translated = _translate_block(verse_raw, src_lang=src_lang, tgt_lang=tgt_lang)
        sections.append(f"[Verse]\n{verse_translated}")
        if chorus_translated and i < len(verses) - 1:
            sections.append(f"[Chorus]\n{chorus_translated}")

    return "\n\n".join(sections)


# ============================================================
# MUSIC GENERATION (AceStep)
# ============================================================
def _generate_music_tensor(lyrics, music_description, duration_secs, seed=42):
    from diffusers import AceStepPipeline
    pipe = AceStepPipeline.from_pretrained(
        MUSIC_MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda:0")
    pipe.vae.enable_tiling()

    output = pipe(
        prompt=f"Fast tempo afrobeats: {music_description}",
        lyrics=lyrics,
        audio_duration=duration_secs,
        generator=torch.Generator(device="cuda:0").manual_seed(seed),
    )
    return output.audios[0]   # (channels, samples) @ 48 kHz


# ============================================================
# UNIFIED ENTRY POINT
# ============================================================
def generate_audio(result, flag, src_lang, tgt_lang, output_audio_file=None):
    """Unified audio pipeline. flag = 'speech' or 'music'."""
    flag = flag.strip().lower()

    if flag == "speech":
        raw_text = result["broadcast_text"]
        audio_tensor, sample_rate = _text_to_audio_tensor(
            raw_text, src_lang=src_lang, tgt_lang=tgt_lang,
        )
        translated_text = _translate_block(
            _clean_text(raw_text), src_lang=src_lang, tgt_lang=tgt_lang,
        )
        if output_audio_file:
            sf.write(output_audio_file, audio_tensor.squeeze().numpy(), sample_rate)
            print(f"\nAudio saved → {output_audio_file}")

        print("\n" + "=" * 60)
        print("FINAL TRANSLATION")
        print("=" * 60)
        print(translated_text)
        return {"audio": audio_tensor, "text": translated_text}

    elif flag == "music":
        lyrics = _format_and_translate_lyrics(result, src_lang=src_lang, tgt_lang=tgt_lang)
        print("\n" + "=" * 60)
        print("TRANSLATED LYRICS")
        print("=" * 60)
        print(lyrics)

        audio_tensor = _generate_music_tensor(
            lyrics=lyrics,
            music_description=result["music_description"],
            duration_secs=result["duration_secs"],
        )
        if output_audio_file:
            audio_np = audio_tensor.cpu().numpy().T   # (samples, channels)
            sf.write(output_audio_file, audio_np, 48_000)
            print(f"\nAudio saved → {output_audio_file}")
        return {"audio": audio_tensor, "lyrics": lyrics}

    else:
        raise ValueError(f"Unknown flag '{flag}'. Expected 'speech' or 'music'.")
