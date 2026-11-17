import numpy as np
import os
import torch
import evaluate
from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from data import get_datasets, get_label_names

MODEL_NAME = "microsoft/layoutlmv3-base"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints/layoutlmv3-cord")

label_names = get_label_names()
id2label = dict(enumerate(label_names))
label2id = {v: k for k, v in id2label.items()}

seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_preds, true_labels = [], []
    for p_row, l_row in zip(preds, labels):
        true_preds.append([id2label[p] for p, l in zip(p_row, l_row) if l != -100])
        true_labels.append([id2label[l] for l in l_row if l != -100])
    r = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": r["overall_precision"],
        "recall": r["overall_recall"],
        "f1": r["overall_f1"],
        "accuracy": r["overall_accuracy"],
    }

def main():
    train_ds, val_ds, test_ds = get_datasets() #Load sets

    #Define the model
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME, id2label=id2label, label2id=label2id)

    #Define the arguments and hyperparameters of the model
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    #Train the model
    trainer.train()

    #Evaluate the model on test set
    print(trainer.evaluate(test_ds, metric_key_prefix="test"))
    #Save the model
    trainer.save_model(f"{OUTPUT_DIR}/best")

if __name__ == "__main__":
    main()