"""
Data loading and inspection for the invoice extraction pipeline.

Loads the data set and provides helpers to inspect features, splits, and label distributions.
"""
from datasets import load_dataset,load_from_disk,Features, Sequence, Value, Array2D, Array3D
import numpy as np
import os
from transformers import LayoutLMv3Processor, LayoutLMv3TokenizerFast
import yaml
from pathlib import Path

#Name of the dataset on hugging face
DATASET_NAME = "Ssunbell/SROIE_layoutlmv2_sequence"

def load_invoice_dataset():
    """Load the full dataset (train/valid/test splits)."""
    ds = load_dataset(DATASET_NAME)
    return ds

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "fields.yaml"

def get_id2label():
    """Maps the id to label"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}

def to_tag(name):
    #seqeval needs BIO-style prefixes to group tokens into entities
    return name if name == "O" else f"I-{name}"

tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base",tokenizer=tokenizer,apply_ocr=False)

MAX_LENGTH = 512 #Number of tokens

def preprocess_example(example):
    """
    Tokenizes the words and makes the necessary changes to boxes , labels
    """
    refactored_image = np.array(example["image"], dtype=np.uint8).transpose(1, 2, 0)       
    encoding = processor(
        images=refactored_image,
        text=example["words"],
        boxes=example["boxes"],
        word_labels=example["labels"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="np",
    )
    return {k: v.squeeze(0) for k, v in encoding.items()}

#The directory where we're saving our processed sets
PROCESSED_DIR = os.environ.get("PROCESSED_DIR","data/processed_512")
#Defines the features we're going to use for our datasets
features = Features({
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "bbox": Array2D(dtype="int64", shape=(MAX_LENGTH, 4)),
    "labels": Sequence(Value("int64")),
    "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
})


def get_datasets():
    """Return preprocessed (train, val, test), using the saved copy if it exists."""
    if os.path.exists(f"{PROCESSED_DIR}/train"):
        train = load_from_disk(f"{PROCESSED_DIR}/train")
        val = load_from_disk(f"{PROCESSED_DIR}/val")
        test = load_from_disk(f"{PROCESSED_DIR}/test")
    else:
        ds = load_invoice_dataset()
        splits = {}
        for name, split in [("train", "train"), ("val", "val"), ("test", "test")]:
            splits[name] = ds[split].map(
                preprocess_example,
                remove_columns=ds[split].column_names,
                features=features,
            )
            splits[name].save_to_disk(f"{PROCESSED_DIR}/{name}")
        train, val, test = splits["train"], splits["val"], splits["test"]

    for d in (train, val, test):
        d.set_format("torch")
    return train, val, test

if __name__ == "__main__":
    ds = load_invoice_dataset()
    example = ds["train"][0]
    refactored_image = np.array(example["image"], dtype=np.uint8).transpose(1, 2, 0)
    print(refactored_image.shape)
    