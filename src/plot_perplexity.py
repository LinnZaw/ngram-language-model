import matplotlib.pyplot as plt

# Model names
models = ["Unigram", "Bigram", "Trigram"]

# Perplexity values from final evaluation
perplexity = [567.47, 719.03, 3454.44]

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(models, perplexity)

# Title and labels
plt.title("Perplexity Comparison")
plt.xlabel("N-gram Model")
plt.ylabel("Perplexity")

# Display values above bars
for bar, value in zip(bars, perplexity):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{value:.2f}",
        ha="center"
    )

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/perplexity_comparison.png",
    dpi=300
)

# Display graph
plt.show()