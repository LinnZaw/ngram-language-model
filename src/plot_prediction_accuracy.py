import matplotlib.pyplot as plt

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Prediction accuracy
top1_accuracy = [5.19, 15.11, 16.57]
top5_accuracy = [18.13, 32.74, 28.36]

# Create figure
plt.figure(figsize=(9, 5))

# Plot lines
plt.plot(
    models,
    top1_accuracy,
    marker="o",
    linewidth=2,
    label="Top-1 Accuracy"
)

plt.plot(
    models,
    top5_accuracy,
    marker="o",
    linewidth=2,
    label="Top-5 Accuracy"
)

# Title and labels
plt.title("Prediction Accuracy Across N-gram Models")
plt.xlabel("N-gram Model")
plt.ylabel("Accuracy (%)")

# Y-axis
plt.ylim(0, 40)

# Add values to points
for model, value in zip(models, top1_accuracy):
    plt.text(
        model,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

for model, value in zip(models, top5_accuracy):
    plt.text(
        model,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

# Legend
plt.legend()

# Layout
plt.tight_layout()

# Save
plt.savefig(
    "results/figures/prediction_accuracy.png",
    dpi=300
)

# Display
plt.show()
