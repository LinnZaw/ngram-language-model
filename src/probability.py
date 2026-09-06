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
print("Training sentences:", len(sentences))


# ============================================================
# 2. CREATE N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sentence in sentences:

    tokens = sentence.strip().split()

    # Unigram
    for token in tokens:
        unigram_counts[token] += 1

    # Bigram
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] += 1

    # Trigram
    for i in range(len(tokens) - 2):
        trigram = (
            tokens[i],
            tokens[i + 1],
            tokens[i + 2]
        )
        trigram_counts[trigram] += 1


# ============================================================
# 3. CALCULATE UNIGRAM MLE
# ============================================================

total_tokens = sum(unigram_counts.values())

unigram_probabilities = {}

for word, count in unigram_counts.items():

    probability = count / total_tokens

    unigram_probabilities[word] = probability


# ============================================================
# 4. CALCULATE BIGRAM MLE
# ============================================================

bigram_probabilities = {}

for (word1, word2), count in bigram_counts.items():

    probability = count / unigram_counts[word1]

    bigram_probabilities[(word1, word2)] = probability


# ============================================================
# 5. CALCULATE TRIGRAM MLE
# ============================================================

trigram_probabilities = {}

for (word1, word2, word3), count in trigram_counts.items():

    probability = count / bigram_counts[(word1, word2)]

    trigram_probabilities[
        (word1, word2, word3)
    ] = probability


# ============================================================
# 6. DISPLAY EXAMPLES
# ============================================================

print()
print("=" * 60)
print("MLE PROBABILITIES")
print("=" * 60)


# -------------------------
# Unigram
# -------------------------

print()
print("UNIGRAM")
print("-" * 60)

for word, probability in list(
    unigram_probabilities.items()
)[:10]:

    print(
        f"P({word}) = {probability:.6f}"
    )


# -------------------------
# Bigram
# -------------------------

print()
print("BIGRAM")
print("-" * 60)

for bigram, probability in list(
    bigram_probabilities.items()
)[:10]:

    print(
        f"P({bigram[1]} | {bigram[0]}) = "
        f"{probability:.6f}"
    )


# -------------------------
# Trigram
# -------------------------

print()
print("TRIGRAM")
print("-" * 60)

for trigram, probability in list(
    trigram_probabilities.items()
)[:10]:

    print(
        f"P({trigram[2]} | {trigram[0]} {trigram[1]}) = "
        f"{probability:.6f}"
    )


# ============================================================
# 7. TEST SPECIFIC EXAMPLES
# ============================================================

print()
print("=" * 60)
print("SPECIFIC EXAMPLES")
print("=" * 60)


# Bigram example
bigram = ("to", "sherlock")

if bigram in bigram_probabilities:

    print()
    print(
        f"P(sherlock | to) = "
        f"{bigram_probabilities[bigram]:.6f}"
    )

    print(
        f"Count(to sherlock) = "
        f"{bigram_counts[bigram]}"
    )

    print(
        f"Count(to) = "
        f"{unigram_counts['to']}"
    )


# Trigram example
trigram = ("to", "sherlock", "holmes")

if trigram in trigram_probabilities:

    print()
    print(
        f"P(holmes | to sherlock) = "
        f"{trigram_probabilities[trigram]:.6f}"
    )

    print(
        f"Count(to sherlock holmes) = "
        f"{trigram_counts[trigram]}"
    )

    print(
        f"Count(to sherlock) = "
        f"{bigram_counts[('to', 'sherlock')]}"
    )