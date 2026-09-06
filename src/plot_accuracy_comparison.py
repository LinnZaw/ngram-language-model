import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Accuracy values from final evaluation
top1 = [5.19, 15.11, 16.57]
top5 = [18.13, 32.74, 28.36]

# X-axis positions
x = np.arange(len(models))

# Bar width
width = 0.35

# Create figure
plt.figure(figsize=(9, 5))

# Create bars
bars1 = plt.bar(x - width / 2, top1, width, label="Top-1 Accuracy")
bars2 = plt.bar(x + width / 2, top5, width, label="Top-5 Accuracy")

# Title and labels
plt.title("Top-1 and Top-5 Accuracy Comparison")
plt.xlabel("N-gram Model")
plt.ylabel("Accuracy (%)")

# X-axis labels
plt.xticks(x, models)

# Y-axis range
plt.ylim(0, 40)

# Display values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.2f}%",
            ha="center"
        )

# Show legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/accuracy_comparison.png",
    dpi=300
)

# Display graph
plt.show()