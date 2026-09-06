import matplotlib.pyplot as plt

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Top-5 accuracy values from final evaluation
accuracy = [18.13, 32.74, 28.36]

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracy)

# Title and labels
plt.title("Top-5 Accuracy Comparison")
plt.xlabel("N-gram Model")
plt.ylabel("Top-5 Accuracy (%)")

# Set Y-axis range
plt.ylim(0, 40)

# Display percentage values above bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.2f}%",
        ha="center"
    )

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/top5_accuracy_comparison.png",
    dpi=300
)

# Display graph
plt.show()