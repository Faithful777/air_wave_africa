# 🌍 RAG Broadcast Generation System

Gemma 4 + Local Knowledge Base → Speech Scripts & Educational Songs.

Generates simple-English radio/TTS content and educational music for rural
sub-Saharan African communities, then translates and synthesises audio
output (speech via MMS-TTS, music via ACE-Step).

---

## Requirements

- Python 3.10+
- NVIDIA GPU (T4 or better recommended)
- ~30 GB free disk (for cached models)

## Installation

```bash
git clone <your-repo-url> rag-broadcast
cd rag-broadcast
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
One-Time Setup (download models + knowledge base)
List supported broadcast languages:

```bash
python setup.py --list-languages
```

Run setup with chosen language code (downloads Gemma 4, MMS-TTS for that
language, ACE-Step, sentence-transformers, SeamlessM4T, and the KB file):

```bash
python setup.py --language lug
```

Available codes: amh, lug, nya, sna, som, swh, yor.
Generate Broadcasts
Each broadcast run takes ~16–20 minutes and produces a speech .wav
and a music .wav per run, plus the translated text/lyrics.

```bash
python run.py --runs 3
```

