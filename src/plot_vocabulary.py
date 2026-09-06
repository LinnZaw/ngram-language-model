import matplotlib.pyplot as plt

# Corpus information
categories = [
    "Total Tokens",
    "Vocabulary Size",
    "Unique Bigrams",
    "Unique Trigrams"
]

values = [
    88670,
    6888,
    40453,
    65968
]

# Create bar chart
plt.figure(figsize=(9, 5))

bars = plt.bar(categories, values)

# Title and labels
plt.title("Corpus Statistics")
plt.xlabel("Corpus Measure")
plt.ylabel("Count")

# Rotate labels
plt.xticks(rotation=15)

# Display values above bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1000,
        f"{value:,}",
        ha="center"
    )

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/corpus_statistics.png",
    dpi=300
)

# Display graph
plt.show()