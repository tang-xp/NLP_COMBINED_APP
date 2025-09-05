import os
import re
import spacy
import pandas as pd
from tqdm import tqdm
import joblib
from collections import (
    Counter,
)  # Make sure this is imported if you're using it for diagnostics

# --- Configuration (Your existing config) ---
BASE_DIR = "data/"
DATA_DIRS = [
    os.path.join(BASE_DIR, "custom_train/"),
    os.path.join(BASE_DIR, "custom_test/"),
]

# Load the spaCy model (Your existing spaCy load)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")


# --- REPLACE YOUR OLD `parse_simple_tml_file` FUNCTION WITH THIS NEW ONE ---
def parse_tml_file_robust(file_path):
    """
    Parses a single .tml file, extracts text and annotations, and converts
    them into spaCy tokens and BIO labels. This version uses char offsets
    more carefully.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_content = f.read()

    text_match = re.search(r"<TEXT>(.*?)</TEXT>", full_content, re.DOTALL)
    if not text_match:
        # print(f"Warning: No <TEXT> block found in {file_path}")
        return []

    raw_text_with_tags = text_match.group(1)

    # Store annotations with their original text, start/end in original string
    annotations = []

    # Use a non-greedy regex to find all EVENT and TIMEX3 tags
    # This also captures the content within the tags
    tag_pattern = re.compile(r"<(EVENT|TIMEX3)([^>]*)>(.*?)</\1>", re.DOTALL)

    for match in tag_pattern.finditer(raw_text_with_tags):
        tag_name = match.group(1)
        attributes = match.group(2)
        entity_text = match.group(3).strip()  # The text content of the entity

        # Calculate character offsets in the *original text with tags*
        start_char_in_tags_string = match.start(
            3
        )  # Start of entity_text in raw_text_with_tags
        end_char_in_tags_string = match.end(
            3
        )  # End of entity_text in raw_text_with_tags

        label_type = None
        if tag_name == "EVENT":
            label_type = "EVENT"
        elif tag_name == "TIMEX3":
            type_match = re.search(r'type="([^"]+)"', attributes)
            if type_match:
                timex_type = type_match.group(1).upper()
                if timex_type == "DATE":
                    label_type = "DATE"
                elif timex_type == "TIME":
                    label_type = "TIME"

        if label_type:
            annotations.append(
                {
                    "text": entity_text,
                    "label": label_type,
                    "start": start_char_in_tags_string,
                    "end": end_char_in_tags_string,
                }
            )

    # Get the clean text by removing all XML tags
    clean_text = re.sub(r"<[^>]+>", "", raw_text_with_tags)

    # Now, process the clean text with spaCy
    doc = nlp(clean_text)

    # Prepare entity spans for spaCy, adjusting offsets to the clean text
    # This is the tricky part: original offsets vs clean text offsets

    # Use spaCy's built-in `doc.ents` for better token alignment, then manually verify
    # For now, we will map labels to tokens AFTER getting clean text.

    # Create an offset mapping to convert original text indices to clean text indices
    offset_map = {}
    clean_text_idx = 0
    raw_text_idx = 0
    while raw_text_idx < len(raw_text_with_tags):
        if raw_text_with_tags[raw_text_idx] == "<":
            # Skip the tag
            while (
                raw_text_idx < len(raw_text_with_tags)
                and raw_text_with_tags[raw_text_idx] != ">"
            ):
                raw_text_idx += 1
            raw_text_idx += 1  # Skip the '>'
        else:
            offset_map[raw_text_idx] = clean_text_idx
            raw_text_idx += 1
            clean_text_idx += 1

    # Initialize BIO labels for all tokens as 'O'
    token_labels = ["O"] * len(doc)

    # Apply labels from annotations
    for annot in annotations:
        # Convert original annotation offsets to clean text offsets
        clean_start_char = offset_map.get(annot["start"], -1)
        # For the end, we need to find the clean text index of the character *just after* the entity
        # If the end of the entity in the raw text was raw_end, its corresponding clean text index is clean_end
        # The offset_map might not directly contain raw_end if it's inside a tag, so we find the closest

        raw_end_char_in_text = annot["end"]
        clean_end_char = -1

        # Find the clean text position corresponding to the end of the raw text entity
        temp_raw_idx = raw_end_char_in_text
        while temp_raw_idx > annot["start"] and temp_raw_idx not in offset_map:
            temp_raw_idx -= 1

        if temp_raw_idx in offset_map:
            clean_end_char = offset_map[temp_raw_idx] + (
                raw_end_char_in_text - temp_raw_idx
            )

            # Now, use doc.char_span to get the tokens that align
            span = doc.char_span(
                clean_start_char, clean_end_char, alignment_mode="contract"
            )

            if span is not None:
                for i, token in enumerate(span):
                    prefix = "B-" if i == 0 else "I-"
                    token_labels[token.i] = f"{prefix}{annot['label']}"
            # else:
            # print(f"  -> WARNING: Failed to map span for '{annot['text']}' ({annot['label']}) in {file_path}")
        # else:
        # print(f"  -> WARNING: Failed to find clean_end_char for '{annot['text']}' in {file_path}")

    processed_sentences = []
    for sent in doc.sents:
        tokens = [token.text for token in sent]
        sent_labels = token_labels[sent.start : sent.end]
        if tokens and len(tokens) == len(sent_labels):  # Basic check for alignment
            processed_sentences.append({"tokens": tokens, "labels": sent_labels})
        # else:
        # print(f"  -> WARNING: Sentence token/label mismatch in {file_path}")

    return processed_sentences


# --- Change the call to use the new robust parser ---
# --- Main Script Execution ---
if __name__ == "__main__":
    # --- START OF CHANGE ---
    # Define where the final .pkl files should be saved.
    OUTPUT_DIR = "xp/"

    # Make sure this output directory exists.
    # The 'exist_ok=True' flag means it won't raise an error if the folder is already there.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # --- END OF CHANGE ---

    for data_dir in DATA_DIRS:
        if not os.path.isdir(data_dir):
            print(f"Warning: The directory '{data_dir}' was not found. Skipping.")
            continue

        print(f"\nSearching for .tml files in '{data_dir}'...")
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".tml"):
                    file_list.append(os.path.join(root, filename))

        print(f"Found {len(file_list)} files to process.")

        all_sentences = []
        for file_path in tqdm(file_list, desc=f"Processing"):
            # CALL THE NEW ROBUST PARSER HERE
            all_sentences.extend(parse_tml_file_robust(file_path))

        processed_data = [(sent["tokens"], sent["labels"]) for sent in all_sentences]

        # Label verification (this part is good, no changes needed)
        print(f"\n--- Label Verification for {data_dir} ---")
        all_labels = [label for sent in processed_data for label in sent[1]]
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common():
            print(f"{label:<15} | {count}")

        # --- START OF CHANGE ---
        # Construct the base filename (e.g., 'xp_train_data.pkl')
        dir_name = os.path.basename(os.path.normpath(data_dir))
        base_filename = "xp_" + dir_name.replace("custom_", "") + "_data.pkl"

        # Create the full, correct output path by joining the directory and filename
        full_output_path = os.path.join(OUTPUT_DIR, base_filename)

        # Save the file to the new, full path
        joblib.dump(processed_data, full_output_path)
        # --- END OF CHANGE ---

        print(f"Processing for {data_dir} complete!")
        print(f"Extracted {len(processed_data)} sentences.")
        # --- START OF CHANGE ---
        # Update the final print message to show the correct path
        print(f"Saved processed data to '{full_output_path}'")