from pathlib import Path
from collections import Counter
import math


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

train_file = Path("data/processed/train.txt")
test_file = Path("data/processed/test.txt")

with open(train_file, "r", encoding="utf-8") as file:
    train_sentences = file.readlines()

with open(test_file, "r", encoding="utf-8") as file:
    test_sentences = file.readlines()


# ============================================================
# 2. BUILD N-GRAM COUNTS
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


total_tokens = sum(unigram_counts.values())
vocabulary_size = len(unigram_counts)

print("=" * 70)
print("FINAL N-GRAM MODEL EVALUATION")
print("=" * 70)

print()
print("Training sentences :", len(train_sentences))
print("Testing sentences  :", len(test_sentences))
print("Training tokens    :", total_tokens)
print("Vocabulary size     :", vocabulary_size)


# ============================================================
# 3. UNIGRAM PROBABILITY
# ============================================================

def unigram_probability(word, alpha=0.1):

    return (
        unigram_counts[word] + alpha
    ) / (
        total_tokens + alpha * vocabulary_size
    )


# ============================================================
# 4. BIGRAM PROBABILITY
# ============================================================

def bigram_probability(
    word1,
    word2,
    alpha=0.1
):

    numerator = (
        bigram_counts[(word1, word2)]
        + alpha
    )

    denominator = (
        unigram_counts[word1]
        + alpha * vocabulary_size
    )

    return numerator / denominator


# ============================================================
# 5. TRIGRAM PROBABILITY
# ============================================================

def trigram_probability(
    word1,
    word2,
    word3,
    alpha=0.1
):

    numerator = (
        trigram_counts[
            (word1, word2, word3)
        ]
        + alpha
    )

    denominator = (
        bigram_counts[
            (word1, word2)
        ]
        + alpha * vocabulary_size
    )

    return numerator / denominator


# ============================================================
# 6. UNIGRAM TOP-K PREDICTION
# ============================================================

def predict_unigram(top_k=5):

    candidates = []

    for word in unigram_counts:

        probability = unigram_probability(word)

        candidates.append(
            (word, probability)
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# 7. BIGRAM TOP-K PREDICTION
# ============================================================

def predict_bigram(
    word,
    top_k=5
):

    candidates = []

    if unigram_counts[word] == 0:
        return []

    for next_word in unigram_counts:

        probability = bigram_probability(
            word,
            next_word
        )

        candidates.append(
            (next_word, probability)
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# 8. TRIGRAM TOP-K PREDICTION
# ============================================================

def predict_trigram(
    word1,
    word2,
    top_k=5
):

    candidates = []

    if bigram_counts[(word1, word2)] == 0:
        return []

    for next_word in unigram_counts:

        probability = trigram_probability(
            word1,
            word2,
            next_word
        )

        candidates.append(
            (next_word, probability)
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# 9. EVALUATE UNIGRAM
# ============================================================

def evaluate_unigram():

    top1_correct = 0
    top5_correct = 0
    total = 0

    predictions = predict_unigram(5)

    predicted_words = [
        word
        for word, probability in predictions
    ]

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for actual_word in tokens:

            total += 1

            if predicted_words[0] == actual_word:
                top1_correct += 1

            if actual_word in predicted_words:
                top5_correct += 1

    top1 = top1_correct / total
    top5 = top5_correct / total

    return top1, top5


# ============================================================
# 10. EVALUATE BIGRAM
# ============================================================

def evaluate_bigram():

    top1_correct = 0
    top5_correct = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for i in range(len(tokens) - 1):

            word1 = tokens[i]
            actual_word = tokens[i + 1]

            predictions = predict_bigram(
                word1,
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

            if predicted_words[0] == actual_word:
                top1_correct += 1

            if actual_word in predicted_words:
                top5_correct += 1

    top1 = top1_correct / total
    top5 = top5_correct / total

    return top1, top5


# ============================================================
# 11. EVALUATE TRIGRAM
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

            if predicted_words[0] == actual_word:
                top1_correct += 1

            if actual_word in predicted_words:
                top5_correct += 1

    top1 = top1_correct / total
    top5 = top5_correct / total

    return top1, top5


# ============================================================
# 12. UNIGRAM PERPLEXITY
# ============================================================

def unigram_perplexity():

    log_probability = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for word in tokens:

            probability = unigram_probability(
                word
            )

            log_probability += math.log(
                probability
            )

            total += 1

    return math.exp(
        -log_probability / total
    )


# ============================================================
# 13. BIGRAM PERPLEXITY
# ============================================================

def bigram_perplexity():

    log_probability = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for i in range(len(tokens) - 1):

            word1 = tokens[i]
            word2 = tokens[i + 1]

            probability = bigram_probability(
                word1,
                word2
            )

            log_probability += math.log(
                probability
            )

            total += 1

    return math.exp(
        -log_probability / total
    )


# ============================================================
# 14. TRIGRAM PERPLEXITY
# ============================================================

def trigram_perplexity():

    log_probability = 0
    total = 0

    for sentence in test_sentences:

        tokens = sentence.strip().split()

        for i in range(len(tokens) - 2):

            word1 = tokens[i]
            word2 = tokens[i + 1]
            word3 = tokens[i + 2]

            probability = trigram_probability(
                word1,
                word2,
                word3
            )

            log_probability += math.log(
                probability
            )

            total += 1

    return math.exp(
        -log_probability / total
    )


# ============================================================
# 15. RUN ALL EVALUATIONS
# ============================================================

print()
print("=" * 70)
print("CALCULATING RESULTS...")
print("=" * 70)

unigram_top1, unigram_top5 = evaluate_unigram()

bigram_top1, bigram_top5 = evaluate_bigram()

trigram_top1, trigram_top5 = evaluate_trigram()

unigram_pp = unigram_perplexity()

bigram_pp = bigram_perplexity()

trigram_pp = trigram_perplexity()


# ============================================================
# 16. FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Model':<12}"
    f"{'Top-1':>12}"
    f"{'Top-5':>12}"
    f"{'Perplexity':>15}"
)

print("-" * 51)

print(
    f"{'Unigram':<12}"
    f"{unigram_top1 * 100:>11.2f}%"
    f"{unigram_top5 * 100:>11.2f}%"
    f"{unigram_pp:>15.2f}"
)

print(
    f"{'Bigram':<12}"
    f"{bigram_top1 * 100:>11.2f}%"
    f"{bigram_top5 * 100:>11.2f}%"
    f"{bigram_pp:>15.2f}"
)

print(
    f"{'Trigram':<12}"
    f"{trigram_top1 * 100:>11.2f}%"
    f"{trigram_top5 * 100:>11.2f}%"
    f"{trigram_pp:>15.2f}"
)


# ============================================================
# 17. BEST MODEL
# ============================================================

models = {
    "Unigram": unigram_top1,
    "Bigram": bigram_top1,
    "Trigram": trigram_top1
}

best_model = max(
    models,
    key=models.get
)

print()
print("=" * 70)
print("BEST TOP-1 MODEL")
print("=" * 70)

print(
    f"{best_model} "
    f"({models[best_model] * 100:.2f}%)"
)