from __future__ import annotations

from pathlib import Path
from datasets import load_from_disk

from mtlab.preprocessing.text import (
    tokenize_english,
    tokenize_hindi 
)

BATCH_SIZE = 8192 
RESERVIOR_SIZE = 20000
RANDOM_SEED = 42

REQUIRED_COLUMNS ={"idx","src","tgt"}

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (ROOT /"data"/"processed"/"samanantar_en_hi")
OUTPUT_DIR = (ROOT/"data"/"processed"/"samanantar"/"en-hi"/"word")

def export_split(split,name):
    english_path = OUTPUT_DIR/f"{name}.en"
    hindi_path = OUTPUT_DIR/f"{name}.hi"
    ids_path= OUTPUT_DIR/f"{name}.id"
    with (
        english_path.open("w",encoding="utf-8") as en_file ,
        hindi_path.open("w",encoding="utf-8") as hi_file,
        ids_path.open("w",encoding="utf-8") as id_file
    ):
        for start in range(0,len(split),BATCH_SIZE):
            batch = split[start:start+BATCH_SIZE]
            for idx , source , target in zip(batch["idx"], batch["src"],batch["tgt"],strict=True):
                en_tokens = tokenize_english(source)
                hi_tokens = tokenize_hindi(target)
                en_file.write(" ".join(en_tokens)+'\n')
                hi_file.write(" ".join(hi_tokens)+"\n")
                id_file.write(str(idx)+"\n")
    print("done")
    
    
def main():
    dataset = load_from_disk(str(DATASET_PATH))
    train = dataset["train"]
    validation = dataset["validation"]
    OUTPUT_DIR.mkdir(parents=True , exist_ok=True)
    export_split(train , "train")
    export_split(validation,"validation")
    print("preprocessing completed")
    
if __name__ =="__main__":
    main()
