from pathlib import Path
import random


# ============================================================
# FILE PATHS
# ============================================================

input_file = Path("data/processed/sherlock_clean.txt")

train_file = Path("data/processed/train.txt")
test_file = Path("data/processed/test.txt")


# ============================================================
# 1. READ PROCESSED CORPUS
# ============================================================

with open(input_file, "r", encoding="utf-8") as file:
    sentences = file.readlines()

print("Processed corpus loaded.")
print("Total sentences:", len(sentences))


# ============================================================
# 2. SHUFFLE SENTENCES
# ============================================================

# Use a fixed seed so that we get the same split every time.

random.seed(42)

random.shuffle(sentences)


# ============================================================
# 3. CALCULATE 80/20 SPLIT
# ============================================================

train_size = int(len(sentences) * 0.80)

train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]


# ============================================================
# 4. SAVE TRAINING DATA
# ============================================================

with open(train_file, "w", encoding="utf-8") as file:
    file.writelines(train_sentences)


# ============================================================
# 5. SAVE TESTING DATA
# ============================================================

with open(test_file, "w", encoding="utf-8") as file:
    file.writelines(test_sentences)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print()
print("=" * 50)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 50)

print()
print(f"Total sentences : {len(sentences):,}")
print(f"Training set    : {len(train_sentences):,}")
print(f"Testing set     : {len(test_sentences):,}")

print()
print("Training file:")
print(train_file)

print()
print("Testing file:")
print(test_file)

print()
print("=" * 50)