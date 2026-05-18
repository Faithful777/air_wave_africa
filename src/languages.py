"""Supported broadcast languages."""

# code -> (display name, region description)
LANGUAGES = {
    "amh": ("Amharic", "Horn of Africa / Ethiopia"),
    "lug": ("Ganda",   "East Africa / Uganda"),
    "nya": ("Nyanja",  "Southeast Africa / Malawi-Zambia region"),
    "sna": ("Shona",   "Southern Africa / Zimbabwe"),
    "som": ("Somali",  "Horn of Africa / Somalia"),
    "swh": ("Swahili", "East African coast / Great Lakes region"),
    "yor": ("Yoruba",  "West Africa / Nigeria-Benin region"),
}


def get_region(code: str) -> str:
    if code not in LANGUAGES:
        raise ValueError(f"Unknown language code '{code}'. "
                         f"Choose from: {', '.join(LANGUAGES)}")
    return LANGUAGES[code][1]
