import matplotlib.pyplot as plt
from collections import Counter

# Read training data
with open(
    "data/processed/train.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()

# Tokenize by whitespace
words = text.split()

# Count word frequencies
word_counts = Counter(words)

# Get 10 most frequent words
top_words = word_counts.most_common(10)

# Separate words and frequencies
words = [item[0] for item in top_words]
frequencies = [item[1] for item in top_words]

# Create bar chart
plt.figure(figsize=(10, 6))

bars = plt.bar(words, frequencies)

# Title and labels
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")

# Display frequency above each bar
for bar, value in zip(bars, frequencies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{value:,}",
        ha="center"
    )

# Rotate word labels
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/figures/top10_word_frequency.png",
    dpi=300
)

# Display graph
plt.show()