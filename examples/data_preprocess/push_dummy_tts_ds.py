from datasets import Dataset, DatasetDict

# ───────────────────────────────────────────
# ➊ Plain-text samples (edit freely)
dummy_train_texts = [
    "Hello, this is a dummy training sentence for TTS.",
    "An example sentence for text-to-speech conversion.",
    "The train runs across the whole country.",
    "Let's keep loving Python code!",
    "Here is the last sentence for dataset testing."
]

dummy_test_texts = [
    "This sentence belongs to the test split.",
    "A sentence for validating the TTS system."
]

# ➋ Build a DatasetDict with train / test splits
train_ds = Dataset.from_dict({"text": dummy_train_texts})
test_ds  = Dataset.from_dict({"text": dummy_test_texts})
ds = DatasetDict(train=train_ds, test=test_ds)

# ➌ Push to the Hub
ds.push_to_hub(
    "Seungyoun/dummy_llasa_tts_text",
    private=False,              # set False if you want it public
    commit_message="Initial dummy TTS text dataset"
)

print("✅ Dummy dataset pushed!")
