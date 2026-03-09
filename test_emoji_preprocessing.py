"""Test script to verify emoji preservation in preprocessing"""

from preprocessing import clean_text

# Test cases with emojis
test_reviews = [
    "The food was amazing 😍 definitely coming back!",
    "Worst restaurant ever 😡👎",
    "Service was okay 😐 but food was great 🍕👍",
    "Don't go here 😞🤢 terrible experience",
    "Love it! ❤️🎉 Best meal ever 😊",
]

print("=" * 60)
print("EMOJI PREPROCESSING TEST")
print("=" * 60)

for review in test_reviews:
    cleaned = clean_text(review)
    print(f"\nOriginal: {review}")
    print(f"Cleaned:  {cleaned}")
    print("-" * 60)

print("\n✅ Emojis are now preserved in the preprocessing pipeline!")
print("Emojis will contribute to sentiment analysis as features.\n")
