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


# ============================================================
# BUILD UNIGRAM COUNTS
# ============================================================

unigram_counts = Counter()

for sentence in train_sentences:
    tokens = sentence.strip().split()

    for token in tokens:
        unigram_counts[token] += 1


total_tokens = sum(unigram_counts.values())
vocabulary_size = len(unigram_counts)


print("Training sentences:", len(train_sentences))
print("Testing sentences :", len(test_sentences))
print("Vocabulary size    :", vocabulary_size)
print("Total training tokens:", total_tokens)


# ============================================================
# UNIGRAM PROBABILITY
# ============================================================

def unigram_probability(word):

    count = unigram_counts[word]

    return count / total_tokens


# ============================================================
# TOP-K PREDICTION
# ============================================================

def predict_unigram(top_k=5):

    candidates = []

    for word, count in unigram_counts.items():

        probability = count / total_tokens

        candidates.append(
            (word, probability)
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# EVALUATE UNIGRAM ACCURACY
# ============================================================

def evaluate_unigram():

    predictions = predict_unigram(top_k=5)

    predicted_words = [
        word
        for word, probability in predictions
    ]

    top1_correct = 0
    top5_correct = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for actual_word in tokens:

            total += 1

            if predicted_words[0] == actual_word:
                top1_correct += 1

            if actual_word in predicted_words:
                top5_correct += 1

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
# UNIGRAM PERPLEXITY
# ============================================================

def calculate_unigram_perplexity():

    log_probability = 0
    count = 0

    alpha = 0.1

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for word in tokens:

            # Lidstone smoothing
            probability = (
                unigram_counts[word] + alpha
            ) / (
                total_tokens
                + alpha * vocabulary_size
            )

            log_probability += math.log(
                probability
            )

            count += 1

    return math.exp(
        -log_probability / count
    )


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 60)
print("UNIGRAM MODEL EVALUATION")
print("=" * 60)

top1_accuracy, top5_accuracy, total = (
    evaluate_unigram()
)

print()
print(
    f"Evaluated tokens: {total}"
)

print(
    f"Top-1 Accuracy : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-5 Accuracy : "
    f"{top5_accuracy * 100:.2f}%"
)

perplexity = calculate_unigram_perplexity()

print(
    f"Perplexity     : "
    f"{perplexity:.2f}"
)


# ============================================================
# SHOW TOP 5 WORDS
# ============================================================

print()
print("=" * 60)
print("TOP 5 UNIGRAM PREDICTIONS")
print("=" * 60)

predictions = predict_unigram(5)

for rank, (word, probability) in enumerate(
    predictions,
    start=1
):

    print(
        f"{rank}. {word:<15}"
        f" P = {probability:.6f}"
    )