import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Original results
top1 = np.array([5.19, 15.11, 16.57])
top5 = np.array([18.13, 32.74, 28.36])
perplexity = np.array([567.47, 719.03, 3454.44])

# Normalize each metric to 0-100
top1_score = (top1 / max(top1)) * 100
top5_score = (top5 / max(top5)) * 100

# Perplexity: lower is better, so invert it
perplexity_score = (min(perplexity) / perplexity) * 100

# X-axis positions
x = np.arange(len(models))
width = 0.25

# Create chart
plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    x - width,
    top1_score,
    width,
    label="Top-1 Accuracy"
)

bars2 = plt.bar(
    x,
    top5_score,
    width,
    label="Top-5 Accuracy"
)

bars3 = plt.bar(
    x + width,
    perplexity_score,
    width,
    label="Perplexity Score"
)

# Title and labels
plt.title("Overall N-gram Model Performance Comparison")
plt.xlabel("N-gram Model")
plt.ylabel("Normalized Performance Score")

# X-axis labels
plt.xticks(x, models)

# Y-axis range
plt.ylim(0, 110)

# Add values above bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            ha="center",
            fontsize=9
        )

# Legend
plt.legend()

# Layout
plt.tight_layout()

# Save
plt.savefig(
    "results/figures/overall_model_comparison.png",
    dpi=300
)

# Display
plt.show()