"""Generate N broadcasts end-to-end (speech + music) from the command line."""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from src.audio      import generate_audio, init_audio
from src.config     import (
    DOMAINS,
    GEMMA_MODEL_ID,
    KB_PATH,
    SOURCE_LANG,
    build_config,
)
from src.generation import generate_broadcast_content, init_gemma
from src.kb         import KnowledgeBaseParser
from src.languages  import LANGUAGES
from src.render     import render_music, render_speech
from src.retriever  import Retriever


CONFIG_FILE = Path("config/user_config.json")


def load_user_config():
    if not CONFIG_FILE.exists():
        raise SystemExit(
            f"Missing {CONFIG_FILE}. Run `python setup.py --language CODE` first."
        )
    cfg = json.loads(CONFIG_FILE.read_text())
    if cfg["target_lang"] not in LANGUAGES:
        raise SystemExit(f"Bad target language in {CONFIG_FILE}: {cfg}")
    return cfg


def load_gemma():
    print(f"Loading {GEMMA_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, local_files_only=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        local_files_only=True,
        quantization_config=bnb,
        device_map="auto",
    )
    print("   ✅ Loaded in 4-bit mode")
    return processor, model


def main():
    ap = argparse.ArgumentParser(
        description="Run the RAG-driven radio broadcast generator."
    )
    ap.add_argument("--runs", "-n", type=int, default=1,
                    help="Number of broadcasts to generate (each ≈16-20 mins).")
    ap.add_argument("--output-dir", "-o", default="output",
                    help="Where to save WAV files and text transcripts.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional seed for reproducible domain selection.")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs must be ≥ 1")
    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # User config
    user_cfg     = load_user_config()
    target_lang  = user_cfg["target_lang"]
    region       = user_cfg["region"]
    print(f"Target language : {target_lang} ({user_cfg['name']})")
    print(f"Region          : {region}")

    # Audio module needs to know which TTS model to load lazily.
    init_audio(target_lang)

    # Load Gemma
    processor, model = load_gemma()
    init_gemma(processor, model)

    # KB + retriever
    print("\n📚 Parsing knowledge base ...")
    chunks    = KnowledgeBaseParser().parse(KB_PATH)
    print("\n🔍 Building semantic retriever ...")
    retriever = Retriever(chunks)
    print("\n✅ Knowledge base ready")

    # Random domain assignment per run
    broadcast_runs = [random.choice(DOMAINS) for _ in range(args.runs)]
    print(f"\n🎯 Broadcast plan ({args.runs} runs): {broadcast_runs}")

    # Run each broadcast
    for i, domain in enumerate(broadcast_runs):
        print("\n" + "#" * 70)
        print(f"# Broadcast {i + 1}/{args.runs}  domain={domain}")
        print("#" * 70)

        cfg_music  = build_config(target_lang, region, "music",  domain)
        cfg_speech = build_config(target_lang, region, "speech", domain)

        result_music  = generate_broadcast_content(cfg_music,  retriever)
        result_speech = generate_broadcast_content(cfg_speech, retriever)

        speech_wav = out_dir / f"translated_speech_{i}.wav"
        music_wav  = out_dir / f"translated_{i}.wav"
        text_out   = out_dir / f"broadcast_{i}.txt"

        # Speech audio
        speech_result = generate_audio(
            result=result_speech, flag="speech",
            src_lang=SOURCE_LANG, tgt_lang=target_lang,
            output_audio_file=str(speech_wav),
        )

        # Music audio
        music_result = generate_audio(
            result=result_music, flag="music",
            src_lang=SOURCE_LANG, tgt_lang=target_lang,
            output_audio_file=str(music_wav),
        )

        # Save plain-text summary
        with text_out.open("w", encoding="utf-8") as f:
            f.write(f"Broadcast {i + 1} / {args.runs}\n")
            f.write(f"Domain : {domain}\n")
            f.write(f"Target : {target_lang} — {region}\n\n")
            f.write(render_speech(result_speech) + "\n\n")
            f.write("TRANSLATED SPEECH\n" + "=" * 60 + "\n")
            f.write(speech_result["text"] + "\n\n")
            f.write(render_music(result_music) + "\n\n")
            f.write("TRANSLATED LYRICS\n" + "=" * 60 + "\n")
            f.write(music_result["lyrics"] + "\n")
        print(f"\n📄 Text summary saved → {text_out}")

    print("\n🎉 All broadcasts complete.")


if __name__ == "__main__":
    main()
