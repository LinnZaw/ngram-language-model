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

    # -------------------------
    # Unigram
    # -------------------------

    for token in tokens:
        unigram_counts[token] += 1

    # -------------------------
    # Bigram
    # -------------------------

    for i in range(len(tokens) - 1):

        bigram = (
            tokens[i],
            tokens[i + 1]
        )

        bigram_counts[bigram] += 1

    # -------------------------
    # Trigram
    # -------------------------

    for i in range(len(tokens) - 2):

        trigram = (
            tokens[i],
            tokens[i + 1],
            tokens[i + 2]
        )

        trigram_counts[trigram] += 1


# ============================================================
# 3. VOCABULARY SIZE
# ============================================================

vocabulary_size = len(unigram_counts)

total_tokens = sum(unigram_counts.values())


print()
print("=" * 60)
print("CORPUS INFORMATION")
print("=" * 60)

print()
print("Total tokens    :", total_tokens)
print("Vocabulary size :", vocabulary_size)
print("Unique bigrams  :", len(bigram_counts))
print("Unique trigrams :", len(trigram_counts))


# ============================================================
# 4. MLE FUNCTIONS
# ============================================================

def unigram_mle(word):
    """
    Calculate MLE probability of a unigram.

    P(w) = Count(w) / Total tokens
    """

    count = unigram_counts[word]

    if total_tokens == 0:
        return 0

    return count / total_tokens


def bigram_mle(word1, word2):
    """
    Calculate MLE probability of a bigram.

    P(w2 | w1) =
    Count(w1, w2) / Count(w1)
    """

    bigram = (word1, word2)

    numerator = bigram_counts[bigram]
    denominator = unigram_counts[word1]

    if denominator == 0:
        return 0

    return numerator / denominator


def trigram_mle(word1, word2, word3):
    """
    Calculate MLE probability of a trigram.

    P(w3 | w1, w2) =
    Count(w1, w2, w3) / Count(w1, w2)
    """

    trigram = (word1, word2, word3)

    numerator = trigram_counts[trigram]
    denominator = bigram_counts[(word1, word2)]

    if denominator == 0:
        return 0

    return numerator / denominator


# ============================================================
# 5. LAPLACE / ADD-1 SMOOTHING
# ============================================================

def bigram_laplace(word1, word2):
    """
    Laplace smoothing for bigrams.

    P(w2 | w1) =
    (Count(w1,w2) + 1)
    /
    (Count(w1) + V)
    """

    bigram = (word1, word2)

    numerator = bigram_counts[bigram] + 1

    denominator = (
        unigram_counts[word1]
        + vocabulary_size
    )

    return numerator / denominator


def trigram_laplace(word1, word2, word3):
    """
    Laplace smoothing for trigrams.

    P(w3 | w1,w2) =
    (Count(w1,w2,w3) + 1)
    /
    (Count(w1,w2) + V)
    """

    trigram = (word1, word2, word3)

    numerator = trigram_counts[trigram] + 1

    denominator = (
        bigram_counts[(word1, word2)]
        + vocabulary_size
    )

    return numerator / denominator


# ============================================================
# 6. LIDSTONE SMOOTHING
# ============================================================

def bigram_lidstone(word1, word2, alpha):
    """
    Lidstone smoothing for bigrams.

    P(w2 | w1) =
    (Count(w1,w2) + alpha)
    /
    (Count(w1) + alpha * V)
    """

    bigram = (word1, word2)

    numerator = (
        bigram_counts[bigram]
        + alpha
    )

    denominator = (
        unigram_counts[word1]
        + alpha * vocabulary_size
    )

    return numerator / denominator


def trigram_lidstone(word1, word2, word3, alpha):
    """
    Lidstone smoothing for trigrams.

    P(w3 | w1,w2) =
    (Count(w1,w2,w3) + alpha)
    /
    (Count(w1,w2) + alpha * V)
    """

    trigram = (word1, word2, word3)

    numerator = (
        trigram_counts[trigram]
        + alpha
    )

    denominator = (
        bigram_counts[(word1, word2)]
        + alpha * vocabulary_size
    )

    return numerator / denominator


# ============================================================
# 7. TEST WORDS
# ============================================================

print()
print("=" * 60)
print("SMOOTHING EXAMPLES")
print("=" * 60)


# ------------------------------------------------------------
# Example 1: Existing bigram
# ------------------------------------------------------------

word1 = "to"
word2 = "sherlock"

print()
print("Example 1: Existing Bigram")
print("-" * 60)

print(f"Bigram: '{word1} {word2}'")

print(
    "Count:",
    bigram_counts[(word1, word2)]
)

print(
    "Count of first word:",
    unigram_counts[word1]
)

print(
    f"MLE:      {bigram_mle(word1, word2):.10f}"
)

print(
    f"Laplace:  {bigram_laplace(word1, word2):.10f}"
)

print(
    f"Lidstone α=0.1: "
    f"{bigram_lidstone(word1, word2, 0.1):.10f}"
)

print(
    f"Lidstone α=0.5: "
    f"{bigram_lidstone(word1, word2, 0.5):.10f}"
)

print(
    f"Lidstone α=1.0: "
    f"{bigram_lidstone(word1, word2, 1.0):.10f}"
)


# ------------------------------------------------------------
# Example 2: Unseen bigram
# ------------------------------------------------------------

word1 = "sherlock"
word2 = "banana"

print()
print("Example 2: Unseen Bigram")
print("-" * 60)

print(f"Bigram: '{word1} {word2}'")

print(
    "Count:",
    bigram_counts[(word1, word2)]
)

print(
    f"MLE:      {bigram_mle(word1, word2):.10f}"
)

print(
    f"Laplace:  {bigram_laplace(word1, word2):.10f}"
)

print(
    f"Lidstone α=0.1: "
    f"{bigram_lidstone(word1, word2, 0.1):.10f}"
)

print(
    f"Lidstone α=0.5: "
    f"{bigram_lidstone(word1, word2, 0.5):.10f}"
)

print(
    f"Lidstone α=1.0: "
    f"{bigram_lidstone(word1, word2, 1.0):.10f}"
)


# ============================================================
# 8. TRIGRAM EXAMPLE
# ============================================================

word1 = "to"
word2 = "sherlock"
word3 = "holmes"

print()
print("=" * 60)
print("TRIGRAM EXAMPLE")
print("=" * 60)

print()
print(
    f"Trigram: '{word1} {word2} {word3}'"
)

print(
    "Count:",
    trigram_counts[(word1, word2, word3)]
)

print(
    "Count of previous bigram:",
    bigram_counts[(word1, word2)]
)

print(
    f"MLE:      "
    f"{trigram_mle(word1, word2, word3):.10f}"
)

print(
    f"Laplace:  "
    f"{trigram_laplace(word1, word2, word3):.10f}"
)

print(
    f"Lidstone α=0.1: "
    f"{trigram_lidstone(word1, word2, word3, 0.1):.10f}"
)

print(
    f"Lidstone α=0.5: "
    f"{trigram_lidstone(word1, word2, word3, 0.5):.10f}"
)

print(
    f"Lidstone α=1.0: "
    f"{trigram_lidstone(word1, word2, word3, 1.0):.10f}"
)


# ============================================================
# 9. SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print()
print("MLE:")
print("  Uses the original frequency directly.")

print()
print("Laplace:")
print("  Adds 1 to every possible N-gram count.")

print()
print("Lidstone:")
print("  Adds a smaller value alpha to every N-gram count.")

print()
print("Alpha values tested:")
print("  α = 0.1")
print("  α = 0.5")
print("  α = 1.0")

print()
print("Note:")
print("Lidstone α = 1.0 is equivalent to Laplace smoothing.")

print()
print("=" * 60)