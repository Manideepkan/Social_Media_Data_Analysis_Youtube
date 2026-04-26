"""
Stage 3: Multilingual Text Preprocessing
Cleans combined_text using language-aware stopwords, regex filters, and tokenization.
Input : yt_stage1_output.csv
Output: yt_stage3_output.csv
"""

import os
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "yt_stage1_output.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "yt_stage3_output.csv")

# ── Stopword dictionaries (embedded — no extra downloads needed) ──────────────
STOPWORDS: dict[str, set] = {
    "en": {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","should","could","may","might","shall",
        "not","no","nor","so","yet","both","either","neither","each","every",
        "all","any","few","more","most","other","some","such","than","that",
        "this","these","those","then","also","just","from","by","as","if",
        "up","out","about","into","through","during","before","after","above",
        "below","between","same","very","its","it","he","she","they","we","i",
        "my","your","his","her","our","their","what","which","who","whom","how",
        "when","where","why","can","cannot","new","like","video","watch",
        "subscribe","channel","youtube","official","full","please",
    },
    "hi": {
        "और","का","की","के","है","हैं","में","को","से","पर","यह","वह","इस",
        "उस","जो","तो","भी","ने","था","थी","हो","कि","एक","लिए","साथ","वे",
        "कर","हम","आप","मैं","वो","अब","कब","क्यों","कहाँ","जब","बहुत",
        "करते","करना","किया","किसी","अपने","उनके","उनकी",
    },
    "te": {
        "మరియు","లో","కు","తో","ఈ","అది","ఉన్న","చేసిన","వారు","నా","మీ",
        "గా","కి","ని","పై","ఒక","ఉంది","అయిన","చేయడం","నుండి","ఇది",
    },
    "ta": {
        "மற்றும்","இல்","உள்","கள்","என்று","இது","அது","அந்த","இந்த",
        "ஆக","ஒரு","என்ற","உள்ளது","வேண்டும்","அவர்","இவர்",
    },
    "kn": {
        "ಮತ್ತು","ಇಲ್ಲಿ","ಅದು","ಇದು","ಒಂದು","ಅವರು","ಮಾಡಿ","ಇವರು","ಆಗಿ",
        "ನಲ್ಲಿ","ಅಥವಾ","ಗೆ","ದು","ಕ್ಕೆ",
    },
    "ml": {
        "ഒരു","ആണ്","ഇത്","അത്","എന്ന","ഈ","ആ","ഉണ്ട്","ചെയ്ത","കൂടി",
        "വരെ","ആയ","ഒരു","പറഞ്ഞ","ഈ",
    },
    "bn": {
        "এবং","এই","এখন","একটি","হয়","থেকে","আছে","সাথে","করে","কিন্তু",
        "সে","তারা","আমি","আমাদের","যে","আর",
    },
    "mr": {
        "आणि","हा","ही","हे","ते","तो","ती","मी","आम्ही","तुम्ही","आहे",
        "होते","करा","एक","काय","कसे","जे","की",
    },
    "gu": {
        "અને","આ","તે","છે","કે","હું","તમે","આપ","એક","ના","ની","ને",
        "માં","પર","સાથે","હતી","હતો","હશે",
    },
    "pa": {
        "ਅਤੇ","ਇਸ","ਉਹ","ਹੈ","ਹਨ","ਕਿ","ਦੇ","ਦੀ","ਨੂੰ","ਵਿੱਚ","ਨਾਲ",
        "ਇੱਕ","ਮੈਂ","ਤੁਸੀਂ","ਅਸੀਂ",
    },
    "ja": {"の","に","は","を","た","が","で","て","と","し","れ","さ","ある","いる","も","する","から","な","こと"},
    "ko": {"이","가","을","를","의","에","는","은","로","으로","도","만","과","와","한","하다","있다"},
    "zh": {"的","了","在","是","我","有","和","就","不","人","都","一","一个","上","也","很","到","说","要","去"},
    "ar": {"في","من","على","إلى","أن","هذا","هذه","ما","كان","لا","مع","هو","هي","أو","كل","عن","قد"},
    "fr": {"le","la","les","un","une","des","et","en","à","de","du","que","qui","est","pour","pas","sur","avec"},
    "de": {"der","die","das","und","in","zu","den","von","mit","ist","des","für","auf","im","dem"},
    "es": {"el","la","los","las","un","una","y","en","a","de","que","es","por","con","no","lo","como"},
    "pt": {"o","a","os","as","um","uma","e","em","de","que","para","com","não","por","uma","se"},
    "ru": {"и","в","на","не","что","это","с","как","к","он","она","они","мы","вы","из","по","от","за"},
    "id": {"yang","dan","di","ini","itu","dengan","ke","dari","tidak","ada","untuk","juga","sudah","bisa"},
    "tr": {"ve","bir","bu","da","de","için","ile","ya","ama","ne","en","gibi","çok","daha","var"},
}


def detect_language_by_unicode(text: str) -> str | None:
    """Fast heuristic: check Unicode block counts."""
    ranges = {
        "hi": (0x0900, 0x097F),  "mr": (0x0900, 0x097F),
        "te": (0x0C00, 0x0C7F),  "ta": (0x0B80, 0x0BFF),
        "kn": (0x0C80, 0x0CFF),  "ml": (0x0D00, 0x0D7F),
        "bn": (0x0980, 0x09FF),  "gu": (0x0A80, 0x0AFF),
        "pa": (0x0A00, 0x0A7F),
    }
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in ranges.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    if counts:
        best = max(counts, key=counts.__getitem__)
        if counts[best] >= 3:
            return best
    return None


def clean_text(text: str, lang: str) -> str:
    """Clean text: remove noise, apply stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove emojis (high Unicode blocks)
    text = re.sub(r"[\U00010000-\U0010FFFF]", " ", text)
    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)
    # Preserve hashtag words, remove #
    text = re.sub(r"#(\w+)", r" \1 ", text)
    # Remove non-alphanumeric except Indic script chars
    text = re.sub(r"[^\w\sऀ-෿一-鿿぀-ヿ؀-ۿ]", " ", text)
    # Remove digits
    text = re.sub(r"\b\d+\b", " ", text)
    # Lowercase Latin chars only
    text = "".join(ch.lower() if ch.isascii() else ch for ch in text)
    # Tokenize
    tokens = text.split()
    # Stopword removal
    stops = STOPWORDS.get(lang, set()) | STOPWORDS.get("en", set())
    tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return " ".join(tokens)


def get_effective_language(row: pd.Series) -> str:
    """Use stored language, fall back to Unicode detection on combined_text."""
    lang = str(row.get("language", "")).strip().lower()
    # Normalise: keep first 2 chars (e.g. 'en-US' → 'en')
    lang = lang[:2] if len(lang) >= 2 else lang
    if lang in ("un", "unknown", "nan", ""):
        detected = detect_language_by_unicode(str(row.get("combined_text", "")))
        lang = detected or "en"
    return lang


def main():
    print("=" * 60)
    print("STAGE 3 — Multilingual Text Preprocessing")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"      {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Effective language ────────────────────────────────────────
    print("\n[2/5] Resolving effective language per row ...")
    df["eff_lang"] = df.apply(get_effective_language, axis=1)
    lang_dist = df["eff_lang"].value_counts()
    print("      Effective language distribution (top 15):")
    for lang, cnt in lang_dist.head(15).items():
        print(f"        {lang:<20} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    # ── Clean text ────────────────────────────────────────────────
    print("\n[3/5] Cleaning text (combined_text) ...")
    df["combined_text"] = df["combined_text"].fillna("")
    df["clean_text"] = [
        clean_text(txt, lang)
        for txt, lang in zip(df["combined_text"], df["eff_lang"])
    ]
    print("      Done.")

    # ── Additional text features ──────────────────────────────────
    print("\n[4/5] Computing post-clean features ...")
    df["clean_word_count"]  = df["clean_text"].str.split().str.len().fillna(0).astype(int)
    df["clean_char_count"]  = df["clean_text"].str.len().fillna(0).astype(int)
    df["unique_word_count"] = df["clean_text"].apply(lambda x: len(set(x.split())) if x else 0)
    df["lexical_diversity"] = np.where(
        df["clean_word_count"] > 0,
        df["unique_word_count"] / df["clean_word_count"],
        0.0,
    )

    # ── Filter out empty texts ────────────────────────────────────
    before = len(df)
    df = df[df["clean_word_count"] >= 2].reset_index(drop=True)
    after = len(df)
    removed = before - after
    print(f"      Removed {removed:,} rows with <2 tokens after cleaning.")
    print(f"      Remaining: {after:,} rows")

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving to {OUTPUT_PATH} ...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Output Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print("\n--- Clean Text Stats ---")
    for col in ["clean_word_count", "clean_char_count", "lexical_diversity"]:
        s = df[col].describe()
        print(f"  {col:<22}: mean={s['mean']:.2f}  median={df[col].median():.2f}  max={s['max']:.2f}")

    print("\n--- Top 30 Most Frequent Tokens (all languages) ---")
    all_tokens = " ".join(df["clean_text"].dropna()).split()
    token_freq = Counter(all_tokens).most_common(30)
    for i, (tok, freq) in enumerate(token_freq, 1):
        print(f"  {i:>2}. {tok:<20} {freq:>8,}")

    print("\n--- Sample Cleaned Texts ---")
    sample = df[["video_id", "eff_lang", "title", "clean_text", "clean_word_count"]].head(5)
    for _, row in sample.iterrows():
        def safe(s): return str(s).encode("ascii", errors="replace").decode("ascii")
        print(f"\n  video_id  : {safe(row['video_id'])}")
        print(f"  lang      : {safe(row['eff_lang'])}")
        print(f"  title     : {safe(str(row['title'])[:80])}")
        print(f"  clean_text: {safe(str(row['clean_text'])[:120])}")
        print(f"  tokens    : {row['clean_word_count']}")

    print("\n[DONE] Stage 3 complete.\n")


if __name__ == "__main__":
    main()
