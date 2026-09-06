from pathlib import Path
from collections import Counter
import math

# ============================================================
# LOAD DATA
# ============================================================

train_file = Path("data/processed/train.txt")
test_file = Path("data/processed/test.txt")

with open(train_file, "r", encoding="utf-8") as file:
    train_sentences = file.readlines()

with open(test_file, "r", encoding="utf-8") as file:
    test_sentences = file.readlines()

print("Training sentences:", len(train_sentences))
print("Testing sentences :", len(test_sentences))


# ============================================================
# BUILD N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sentence in train_sentences:

    tokens = sentence.strip().split()

    for token in tokens:
        unigram_counts[token] += 1

    for i in range(len(tokens) - 1):
        bigram_counts[
            (tokens[i], tokens[i + 1])
        ] += 1

    for i in range(len(tokens) - 2):
        trigram_counts[
            (tokens[i], tokens[i + 1], tokens[i + 2])
        ] += 1


vocabulary_size = len(unigram_counts)

print("Vocabulary size:", vocabulary_size)


# ============================================================
# TRIGRAM MLE PROBABILITY
# ============================================================

def trigram_mle(word1, word2, word3):

    numerator = trigram_counts[
        (word1, word2, word3)
    ]

    denominator = bigram_counts[
        (word1, word2)
    ]

    if denominator == 0:
        return 0

    return numerator / denominator


# ============================================================
# TRIGRAM LIDSTONE PROBABILITY
# ============================================================

def trigram_lidstone(
    word1,
    word2,
    word3,
    alpha=0.1
):

    numerator = (
        trigram_counts[
            (word1, word2, word3)
        ] + alpha
    )

    denominator = (
        bigram_counts[
            (word1, word2)
        ] + alpha * vocabulary_size
    )

    return numerator / denominator


# ============================================================
# TRIGRAM TOP-K PREDICTION
# ============================================================

def predict_trigram(
    word1,
    word2,
    top_k=5
):

    candidates = []

    denominator = bigram_counts[
        (word1, word2)
    ]

    if denominator == 0:
        return []

    for next_word in unigram_counts:

        probability = trigram_mle(
            word1,
            word2,
            next_word
        )

        if probability > 0:

            candidates.append(
                (next_word, probability)
            )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# EVALUATE TRIGRAM ACCURACY
# ============================================================

def evaluate_trigram():

    top1_correct = 0
    top5_correct = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for i in range(len(tokens) - 2):

            word1 = tokens[i]
            word2 = tokens[i + 1]
            actual_word = tokens[i + 2]

            predictions = predict_trigram(
                word1,
                word2,
                top_k=5
            )

            if not predictions:
                continue

            predicted_words = [
                word
                for word, probability
                in predictions
            ]

            total += 1

            # Top-1
            if predicted_words[0] == actual_word:
                top1_correct += 1

            # Top-5
            if actual_word in predicted_words:
                top5_correct += 1

    if total == 0:
        return 0, 0, 0

    top1_accuracy = (
        top1_correct / total
    )

    top5_accuracy = (
        top5_correct / total
    )

    return (
        top1_accuracy,
        top5_accuracy,
        total
    )


# ============================================================
# TRIGRAM PERPLEXITY
# ============================================================

def calculate_trigram_perplexity():

    log_probability = 0
    count = 0

    alpha = 0.1

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for i in range(len(tokens) - 2):

            word1 = tokens[i]
            word2 = tokens[i + 1]
            word3 = tokens[i + 2]

            probability = trigram_lidstone(
                word1,
                word2,
                word3,
                alpha
            )

            log_probability += math.log(
                probability
            )

            count += 1

    if count == 0:
        return float("inf")

    return math.exp(
        -log_probability / count
    )


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 60)
print("TRIGRAM MODEL EVALUATION")
print("=" * 60)

top1_accuracy, top5_accuracy, total = (
    evaluate_trigram()
)

print()
print(f"Evaluated predictions: {total}")

print(
    f"Top-1 Accuracy : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-5 Accuracy : "
    f"{top5_accuracy * 100:.2f}%"
)

perplexity = calculate_trigram_perplexity()

print(
    f"Perplexity     : "
    f"{perplexity:.2f}"
)