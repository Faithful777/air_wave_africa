"""One-time setup: downloads all required models + the knowledge base file."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from src.languages import LANGUAGES
from src.config    import (
    GEMMA_MODEL_ID,
    EMBED_MODEL_ID,
    MUSIC_MODEL_ID,
    TRANSLATION_MODEL_NAME,
    tts_model_name,
)


CONFIG_DIR  = Path("config")
CONFIG_FILE = CONFIG_DIR / "user_config.json"
DATA_DIR    = Path("data")
KB_FILE     = DATA_DIR / "SSA_RAG_KnowledgeBase_Full.txt"
KB_GDRIVE_URL = "https://drive.google.com/file/d/1cbYMkQ0Leufj2Vy3pOK-FLyJG7Rf7G-Z/view?usp=sharing"


def list_languages():
    print("Supported broadcast languages:\n")
    print(f"  {'code':6s}  {'name':10s}  region")
    print(f"  {'-'*6}  {'-'*10}  {'-'*40}")
    for code, (name, region) in LANGUAGES.items():
        print(f"  {code:6s}  {name:10s}  {region}")


def download_models(target_lang: str):
    print(f"\n📦 Downloading models (target language: {target_lang})\n")

    print(f"  • Gemma 4 ({GEMMA_MODEL_ID})")
    snapshot_download(repo_id=GEMMA_MODEL_ID)

    tts_id = tts_model_name(target_lang)
    print(f"  • MMS-TTS ({tts_id})")
    snapshot_download(repo_id=tts_id)

    print(f"  • Music ACE-Step ({MUSIC_MODEL_ID})")
    snapshot_download(repo_id=MUSIC_MODEL_ID)

    print(f"  • Embeddings ({EMBED_MODEL_ID})")
    snapshot_download(repo_id=EMBED_MODEL_ID)

    print(f"  • Translation ({TRANSLATION_MODEL_NAME})")
    snapshot_download(repo_id=TRANSLATION_MODEL_NAME)

    print("\n✅ All models downloaded.")


def download_kb():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if KB_FILE.exists():
        print(f"\n📚 Knowledge base already present at {KB_FILE}")
        return
    print(f"\n📚 Downloading knowledge base → {KB_FILE}")
    import gdown
    gdown.download(url=KB_GDRIVE_URL, output=str(KB_FILE), fuzzy=True, quiet=False)


def save_user_config(target_lang: str):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    name, region = LANGUAGES[target_lang]
    payload = {"target_lang": target_lang, "name": name, "region": region}
    CONFIG_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\n💾 Saved user config → {CONFIG_FILE}")
    print(f"   {payload}")


def main():
    ap = argparse.ArgumentParser(
        description="One-time setup for the RAG Broadcast system."
    )
    ap.add_argument("--language", "-l",
                    help=f"Target broadcast language code "
                         f"({', '.join(LANGUAGES)})")
    ap.add_argument("--list-languages", action="store_true",
                    help="List supported broadcast languages and exit.")
    ap.add_argument("--skip-models", action="store_true",
                    help="Skip the (large) model downloads.")
    ap.add_argument("--skip-kb",     action="store_true",
                    help="Skip downloading the knowledge base text file.")
    args = ap.parse_args()

    if args.list_languages:
        list_languages()
        return

    if not args.language:
        ap.error("You must pass --language CODE (or use --list-languages).")
    if args.language not in LANGUAGES:
        ap.error(f"Unknown language code '{args.language}'. "
                 f"Choose from: {', '.join(LANGUAGES)}")

    if not args.skip_kb:
        download_kb()
    if not args.skip_models:
        download_models(args.language)
    save_user_config(args.language)

    print("\n🎉 Setup complete. Run a broadcast with:")
    print(f"   python run.py --runs 1")


if __name__ == "__main__":
    main()
