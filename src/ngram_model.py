from pathlib import Path
from collections import Counter


# ============================================================
# FILE PATH
# ============================================================

train_file = Path("data/processed/train.txt")


# ============================================================
# 1. READ TRAINING DATA
# ============================================================

with open(train_file, "r", encoding="utf-8") as file:
    sentences = file.readlines()

print("Training data loaded.")
print("Number of training sentences:", len(sentences))


# ============================================================
# 2. CREATE N-GRAM COUNTERS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()


# ============================================================
# 3. GENERATE N-GRAMS
# ============================================================

for sentence in sentences:

    # Convert sentence into a list of words/tokens
    tokens = sentence.strip().split()

    # -------------------------
    # Unigrams
    # -------------------------

    for token in tokens:
        unigram_counts[token] += 1

    # -------------------------
    # Bigrams
    # -------------------------

    for i in range(len(tokens) - 1):

        bigram = (tokens[i], tokens[i + 1])

        bigram_counts[bigram] += 1

    # -------------------------
    # Trigrams
    # -------------------------

    for i in range(len(tokens) - 2):

        trigram = (
            tokens[i],
            tokens[i + 1],
            tokens[i + 2]
        )

        trigram_counts[trigram] += 1


# ============================================================
# 4. DISPLAY STATISTICS
# ============================================================

print()
print("=" * 60)
print("N-GRAM COUNTS")
print("=" * 60)

print()
print("Unique unigrams :", len(unigram_counts))
print("Unique bigrams  :", len(bigram_counts))
print("Unique trigrams :", len(trigram_counts))


# ============================================================
# 5. DISPLAY MOST COMMON N-GRAMS
# ============================================================

print()
print("Top 20 Unigrams")
print("-" * 40)

for word, count in unigram_counts.most_common(20):
    print(f"{word:15} {count}")


print()
print("Top 20 Bigrams")
print("-" * 40)

for bigram, count in bigram_counts.most_common(20):
    print(f"{' '.join(bigram):30} {count}")


print()
print("Top 20 Trigrams")
print("-" * 40)

for trigram, count in trigram_counts.most_common(20):
    print(f"{' '.join(trigram):40} {count}")


# ============================================================
# 6. EXAMPLE COUNTS
# ============================================================

print()
print("=" * 60)
print("EXAMPLE N-GRAM COUNTS")
print("=" * 60)

example_bigram = ("to", "sherlock")
example_trigram = ("to", "sherlock", "holmes")

print()
print(f"Count of 'to sherlock': {bigram_counts[example_bigram]}")
print(f"Count of 'to sherlock holmes': {trigram_counts[example_trigram]}")