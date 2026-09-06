import matplotlib.pyplot as plt

# Dataset sizes
datasets = ["Training", "Testing"]
sentences = [5454, 1364]

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(datasets, sentences)

# Title and labels
plt.title("Training and Testing Data Distribution")
plt.xlabel("Dataset")
plt.ylabel("Number of Sentences")

# Display values above bars
for bar, value in zip(bars, sentences):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{value:,}",
        ha="center"
    )

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/train_test_distribution.png",
    dpi=300
)

# Display graph
plt.show()