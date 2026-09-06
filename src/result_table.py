import pandas as pd

# Final evaluation results
results = {
    "Model": ["Unigram", "Bigram", "Trigram"],
    "Top-1 Accuracy": ["5.19%", "15.11%", "16.57%"],
    "Top-5 Accuracy": ["18.13%", "32.74%", "28.36%"],
    "Perplexity": [567.47, 719.03, 3454.44]
}

# Create DataFrame
df = pd.DataFrame(results)

# Display table
print("\nFINAL N-GRAM MODEL COMPARISON")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

# Save as CSV
df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("results/model_comparison.csv")