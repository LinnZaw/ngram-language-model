from pathlib import Path

file_path = Path("data/raw/Sherlock_Holmes.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Number of characters:", len(text))
print("\nFirst 1000 characters:\n")
print(text[:1000])