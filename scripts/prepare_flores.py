from pathlib import Path
from datasets import load_dataset
from mtlab.preprocessing.text import (
    tokenize_hindi,
    tokenize_english,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (ROOT/"data"/"evaluation"/"flores200"/"en-hi")

DATASET_NAME ="facebook/flores"

def prepare_split(split_name):
    english = load_dataset( DATASET_NAME , "eng_Latn",split=split_name,)
    hindi = load_dataset(DATASET_NAME,"hin_Deva", split = split_name)
    hindi_by_id = {
        row["id"] : row["sentence"] for row in hindi
    }
    raw_dir = OUTPUT_DIR/"raw"
    word_dir = OUTPUT_DIR/"word"
    raw_dir.mkdir(parents=True,exist_ok=True)
    word_dir.mkdir(parents=True, exist_ok = True)
    with (
        (raw_dir/f"{split_name}.en").open("w",encoding="utf-8")as raw_en,
        (raw_dir/f"{split_name}.hi").open("w",encoding="utf-8") as raw_hi,
        (raw_dir/f"{split_name}.ids").open("w",encoding = "utf-8") as raw_ids,
        (word_dir/f"{split_name}.en").open("w",encoding="utf-8") as word_en,
        (word_dir/f"{split_name}.hi").open("w",encoding="utf-8") as word_hi,
        (word_dir/f"{split_name}.ids").open("w",encoding="utf-8") as word_ids
    ):
        for row in english:
            sentence_id = row["id"]
            source =row["sentence"]
            target = hindi_by_id[sentence_id]
            raw_en.write(source+"\n")
            raw_hi.write(target + "\n")
            raw_ids.write(str(sentence_id)+"\n")
            
            word_en.write(" ".join(tokenize_english(source))+"\n")
            word_hi.write(" ".join(tokenize_hindi(target)) +"\n")
            word_ids.write(str(sentence_id)+"\n")
def main():
    prepare_split("dev")
    prepare_split("devtest")
    print("done")
if __name__ == "__main__":
    main()
            