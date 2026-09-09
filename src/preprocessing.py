from pathlib import Path
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")


# ============================================================
# FILE PATHS
# ============================================================

input_file = Path("data/raw/sherlock_holmes.txt")
output_file = Path("data/processed/sherlock_clean.txt")


# ============================================================
# 1. READ RAW CORPUS
# ============================================================

if not input_file.exists():
    print("Error: Raw corpus file not found.")
    print(f"Expected file: {input_file}")
    exit()

with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

print("Raw corpus loaded.")
print("Original characters:", len(text))


# ============================================================
# 2. REMOVE PROJECT GUTENBERG INTRODUCTION
# ============================================================

# The actual first story starts with:
# "To Sherlock Holmes she is always _the_ woman."

start_marker = "To Sherlock Holmes she is always"

start_position = text.find(start_marker)

if start_position != -1:
    text = text[start_position:]
    print("Gutenberg introduction removed.")
else:
    print("Warning: Story start marker not found.")


# ============================================================
# 3. REMOVE PROJECT GUTENBERG FOOTER
# ============================================================

end_markers = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK"
]

footer_removed = False

for marker in end_markers:

    end_position = text.find(marker)

    if end_position != -1:
        text = text[:end_position]
        footer_removed = True
        break

if footer_removed:
    print("Gutenberg footer removed.")
else:
    print("Warning: Gutenberg footer not found.")


# ============================================================
# 4. REMOVE ITALIC MARKERS
# ============================================================


text = re.sub(r"_([^_]+)_", r"\1", text)


# ============================================================
# 5. NORMALIZE WHITESPACE
# ============================================================

text = re.sub(r"\s+", " ", text)

text = text.strip()


# ============================================================
# 6. CONVERT TO LOWERCASE
# ============================================================

text = text.lower()


# ============================================================
# 7. SENTENCE TOKENIZATION
# ============================================================

sentences = sent_tokenize(text)

print("Sentence tokenization completed.")
print("Detected sentences:", len(sentences))


# ============================================================
# 8. WORD TOKENIZATION
# ============================================================

tokenized_sentences = []

for sentence in sentences:

    tokens = word_tokenize(sentence)

    cleaned_tokens = []

    for token in tokens:

        # Keep alphabetic words
        if token.isalpha():
            cleaned_tokens.append(token)

        # Keep basic sentence punctuation
        elif token in [".", "!", "?"]:
            cleaned_tokens.append(token)

    # Only keep non-empty sentences
    if cleaned_tokens:
        tokenized_sentences.append(cleaned_tokens)


# ============================================================
# 9. SAVE PROCESSED CORPUS
# ============================================================

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as file:

    for tokens in tokenized_sentences:
        file.write(" ".join(tokens) + "\n")


# ============================================================
# 10. CALCULATE STATISTICS
# ============================================================

number_of_sentences = len(tokenized_sentences)

number_of_tokens = sum(
    len(sentence)
    for sentence in tokenized_sentences
)

number_of_words = sum(
    1
    for sentence in tokenized_sentences
    for token in sentence
    if token.isalpha()
)

unique_words = set()

for sentence in tokenized_sentences:

    for token in sentence:

        if token.isalpha():
            unique_words.add(token)

vocabulary_size = len(unique_words)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Corpus statistics:")
print(f"Original characters : {len(text):,}")
print(f"Sentences           : {number_of_sentences:,}")
print(f"Words               : {number_of_words:,}")
print(f"Total tokens        : {number_of_tokens:,}")
print(f"Vocabulary size     : {vocabulary_size:,}")

print()
print("Processed file:")
print(output_file)

print()
print("First 5 processed sentences:")
print("-" * 60)

for sentence in tokenized_sentences[:5]:

    print(" ".join(sentence))

print()
print("=" * 60)
print("Done!")
print("=" * 60)