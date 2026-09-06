from pathlib import Path
from collections import Counter

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_file = Path("data/processed/train.txt")

with open(train_file, "r", encoding="utf-8") as file:
    sentences = file.readlines()

print("Training data loaded.")
print("Training sentences:", len(sentences))


# ============================================================
# BUILD N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sentence in sentences:
    tokens = sentence.strip().split()

    # Unigrams
    for token in tokens:
        unigram_counts[token] += 1

    # Bigrams
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] += 1

    # Trigrams
    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
        trigram_counts[trigram] += 1


vocabulary = list(unigram_counts.keys())
vocabulary_size = len(vocabulary)


# ============================================================
# BIGRAM NEXT-WORD PREDICTION
# ============================================================

def predict_bigram(word, top_n=5):
    """
    Predict the next word using a Bigram model.

    P(next_word | word)
    """

    candidates = []

    for (first_word, next_word), count in bigram_counts.items():

        if first_word == word:

            probability = count / unigram_counts[word]

            candidates.append(
                (next_word, probability, count)
            )

    # Sort by probability, highest first
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_n]


# ============================================================
# TRIGRAM NEXT-WORD PREDICTION
# ============================================================

def predict_trigram(word1, word2, top_n=5):
    """
    Predict the next word using a Trigram model.

    P(next_word | word1, word2)
    """

    candidates = []

    previous_bigram = (word1, word2)

    denominator = bigram_counts[previous_bigram]

    if denominator == 0:
        return []

    for (first_word, second_word, next_word), count in trigram_counts.items():

        if first_word == word1 and second_word == word2:

            probability = count / denominator

            candidates.append(
                (next_word, probability, count)
            )

    # Sort by probability, highest first
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_n]


# ============================================================
# DISPLAY BIGRAM PREDICTION
# ============================================================

print()
print("=" * 60)
print("BIGRAM NEXT-WORD PREDICTION")
print("=" * 60)

word = "sherlock"

print()
print(f"Input word: '{word}'")
print()
print("Top predictions:")

predictions = predict_bigram(word, top_n=5)

for rank, (next_word, probability, count) in enumerate(predictions, start=1):

    print(
        f"{rank}. {next_word:<15} "
        f"Probability: {probability:.6f} "
        f"Count: {count}"
    )


# ============================================================
# DISPLAY TRIGRAM PREDICTION
# ============================================================

print()
print("=" * 60)
print("TRIGRAM NEXT-WORD PREDICTION")
print("=" * 60)

word1 = "sherlock"
word2 = "holmes"

print()
print(f"Input words: '{word1} {word2}'")
print()
print("Top predictions:")

predictions = predict_trigram(word1, word2, top_n=5)

for rank, (next_word, probability, count) in enumerate(predictions, start=1):

    print(
        f"{rank}. {next_word:<15} "
        f"Probability: {probability:.6f} "
        f"Count: {count}"
    )


# ============================================================
# UNSEEN INPUT TEST
# ============================================================

print()
print("=" * 60)
print("UNSEEN INPUT TEST")
print("=" * 60)

word1 = "sherlock"
word2 = "banana"

predictions = predict_trigram(word1, word2, top_n=5)

print()
print(f"Input words: '{word1} {word2}'")

if not predictions:
    print("No prediction found.")
else:
    for rank, (next_word, probability, count) in enumerate(predictions, start=1):

        print(
            f"{rank}. {next_word:<15} "
            f"Probability: {probability:.6f} "
            f"Count: {count}"
        )