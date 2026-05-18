"""Global constants and per-run config builder."""

# ---- Model IDs ----
GEMMA_MODEL_ID  = "google/gemma-4-E4B-it"
EMBED_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
MUSIC_MODEL_ID  = "ACE-Step/acestep-v15-xl-turbo-diffusers"

# ---- Translation / TTS ----
SOURCE_LANG               = "eng"
SENTENCE_PAUSE_SECONDS    = 0.45
TRANSLATION_MODEL_NAME    = "facebook/seamless-m4t-v2-large"

def tts_model_name(target_lang: str) -> str:
    return f"facebook/mms-tts-{target_lang}"

# ---- Tuning ----
WORDS_PER_SECOND = 2.5
LYRIC_DENSITY    = 1.4

# ---- Domains ----
DOMAINS = ["health", "agriculture", "education", "economic", "conservation"]

DOMAIN_CHUNK_PREFIX = {
    "health":       "H-",
    "agriculture":  "A-",
    "education":    "E-",
    "economic":     "EC-",
    "conservation": "C-",
}

KB_PATH = "data/SSA_RAG_KnowledgeBase_Full.txt"


def build_config(target_lang_code: str, region: str, mode: str, domain: str) -> dict:
    """Build the per-broadcast configuration block."""
    CONFIG = {
        "mode":   mode,                    # 'speech' or 'music'
        "domain": domain,                  # health|agriculture|...
        "region": f"rural {region}",
        "target_lang": target_lang_code,

        # Speech settings
        "total_words":     2500,
        "top_k":           5,
        "add_transitions": True,

        # Music settings
        "duration_secs":   120,

        # Model
        "load_in_4bit":    True,
    }

    assert CONFIG["domain"] in DOMAINS
    if CONFIG["mode"] == "speech":
        assert CONFIG["total_words"] > 0
        words_per_seg = CONFIG["total_words"] // CONFIG["top_k"]
        print(f"  Speech plan: {CONFIG['top_k']} chunks × "
              f"{words_per_seg} words = "
              f"~{words_per_seg * CONFIG['top_k']} total words")

    print("\n✅ Config validated:")
    for k, v in CONFIG.items():
        print(f"   {k:20s}: {v}")
    return CONFIG
