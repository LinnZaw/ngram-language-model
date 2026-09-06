import matplotlib.pyplot as plt

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Top-1 accuracy values from your final evaluation
accuracy = [5.19, 15.11, 16.57]

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracy)

# Title and labels
plt.title("Top-1 Accuracy Comparison")
plt.xlabel("N-gram Model")
plt.ylabel("Top-1 Accuracy (%)")

# Set Y-axis range
plt.ylim(0, 20)

# Display percentage values above bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}%",
        ha="center"
    )

# Adjust layout
plt.tight_layout()

# Save the graph
plt.savefig(
    "results/figures/top1_accuracy_comparison.png",
    dpi=300
)

# Display graph
plt.show()