from datasets import load_dataset , DatasetDict
from pathlib import Path


dataset = load_dataset("ai4bharat/samanantar","hi",split="train")
print(dataset)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT/"data"/"processed"/"samanantar_en_hi"

dataset = dataset.filter(
    lambda e:(
        bool(e["src"] and e["src"].strip()) and bool(e["tgt"] and e["tgt"].strip())
    ),
    desc="removing empty sentence pairs",
)

splits = dataset.train_test_split(
    test_size =0.02,seed=42,shuffle=True
)

prepared_dataset = DatasetDict({
    "train" : splits["train"],
    "validation":splits["test"]
})

prepared_dataset.save_to_disk(str(OUTPUT_DIR))
print("saved")