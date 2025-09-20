import os, re, joblib, spacy
from tqdm import tqdm
from collections import Counter

BASE_DIR = "data/"
DATA_DIRS = [
    os.path.join(BASE_DIR, "custom_train/"),
    os.path.join(BASE_DIR, "custom_test/"),
]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")


def parse_tml_file_robust(file_path):
    """Parse .tml file into token/BIO label pairs using char offsets."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_content = f.read()

    text_match = re.search(r"<TEXT>(.*?)</TEXT>", full_content, re.DOTALL)
    if not text_match:
        return []

    raw_text_with_tags = text_match.group(1)
    annotations = []
    tag_pattern = re.compile(r"<(EVENT|TIMEX3)([^>]*)>(.*?)</\1>", re.DOTALL)

    for m in tag_pattern.finditer(raw_text_with_tags):
        tag_name, attributes, entity_text = m.group(1), m.group(2), m.group(3).strip()
        start_char, end_char = m.start(3), m.end(3)

        label_type = None
        if tag_name == "EVENT":
            label_type = "EVENT"
        elif tag_name == "TIMEX3":
            t = re.search(r'type="([^"]+)"', attributes)
            if t:
                t = t.group(1).upper()
                label_type = "DATE" if t == "DATE" else "TIME" if t == "TIME" else None

        if label_type:
            annotations.append({
                "text": entity_text, "label": label_type,
                "start": start_char, "end": end_char
            })

    clean_text = re.sub(r"<[^>]+>", "", raw_text_with_tags)
    doc = nlp(clean_text)

    # FIXED: More robust character offset mapping
    offset_map = {}
    ci, ri = 0, 0
    
    while ri < len(raw_text_with_tags) and ci < len(clean_text):
        if raw_text_with_tags[ri] == "<":
            # Skip entire tag
            while ri < len(raw_text_with_tags) and raw_text_with_tags[ri] != ">":
                ri += 1
            if ri < len(raw_text_with_tags):
                ri += 1  # Skip the '>'
        else:
            # Map raw position to clean position
            offset_map[ri] = ci
            ri += 1
            ci += 1

    token_labels = ["O"] * len(doc)
    
    for a in annotations:
        # FIXED: Better handling of character span mapping
        clean_start = offset_map.get(a["start"], None)
        
        # Find clean_end by looking for the last mapped character within the span
        clean_end = None
        for pos in range(a["end"] - 1, a["start"] - 1, -1):
            if pos in offset_map:
                clean_end = offset_map[pos] + 1  # +1 for inclusive end
                break
        
        if clean_start is not None and clean_end is not None:
            # FIXED: Try multiple alignment modes for better coverage
            span = None
            for alignment_mode in ["expand", "contract", "strict"]:
                span = doc.char_span(clean_start, clean_end, alignment_mode=alignment_mode)
                if span:
                    break
            
            if span:
                # Label all tokens in the span
                for i, token in enumerate(span):
                    if token.i < len(token_labels):
                        token_labels[token.i] = f"{'B' if i == 0 else 'I'}-{a['label']}"
            else:
                # FALLBACK: Manual token matching if char_span fails
                entity_words = a["text"].lower().split()
                doc_tokens = [t.text.lower() for t in doc]
                
                # Find the sequence in tokens
                for start_idx in range(len(doc_tokens) - len(entity_words) + 1):
                    if doc_tokens[start_idx:start_idx + len(entity_words)] == entity_words:
                        for i in range(len(entity_words)):
                            token_idx = start_idx + i
                            if token_idx < len(token_labels):
                                token_labels[token_idx] = f"{'B' if i == 0 else 'I'}-{a['label']}"
                        break

    out = []
    for sent in doc.sents:
        toks = [t.text for t in sent]
        labs = token_labels[sent.start:sent.end]
        if toks and len(toks) == len(labs):
            out.append({"tokens": toks, "labels": labs})
    return out


if __name__ == "__main__":
    OUTPUT_DIR = "xp/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for d in DATA_DIRS:
        if not os.path.isdir(d):
            print(f"Warning: Missing dir {d}")
            continue

        print(f"\nScanning {d} ...")
        files = [os.path.join(r, f)
                 for r, _, fs in os.walk(d) for f in fs if f.endswith(".tml")]
        print(f"Found {len(files)} files.")

        all_sents = []
        for fp in tqdm(files, desc="Processing"):
            all_sents.extend(parse_tml_file_robust(fp))

        data = [(s["tokens"], s["labels"]) for s in all_sents]

        print(f"\n--- Label Verification for {d} ---")
        counts = Counter(l for _, labs in data for l in labs)
        for lbl, cnt in counts.most_common():
            print(f"{lbl:<12} | {cnt}")

        fname = "xp_" + os.path.basename(os.path.normpath(d)).replace("custom_", "") + "_data.pkl"
        out_path = os.path.join(OUTPUT_DIR, fname)
        joblib.dump(data, out_path)
        print(f"Saved {len(data)} sentences to {out_path}")