from src.data import to_tag,get_datasets,get_id2label
import torch
from pathlib import Path
import re
import json
from seqeval.metrics import classification_report
from transformers import LayoutLMv3ForTokenClassification
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "layoutlmv3-best"
OUT_DIR = ROOT / "results"

HEADLINE_MIN_PROB = 0.05
SWEEP = (0.0, 0.05, 0.1, 0.2)


def is_total_tag(tag):
    """Checks if a tag ends with exacly TOTAL, so eliminates SUBTOTAL"""
    return tag.split("-")[-1] == "TOTAL"

def word_level_predictions(model, row, id2label, device):
    inputs = {
        k: row[k].unsqueeze(0).to(device)
        for k in ("input_ids", "attention_mask", "bbox", "pixel_values")
    }
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits.float(), dim=-1)
    pred_ids = probs.argmax(-1).cpu().numpy()
    confs = probs.max(-1).values.cpu().numpy()
    labels = row["labels"].cpu().numpy()
    keep = labels != -100

    # NEW: probability of TOTAL, summed over every label id that maps to TOTAL
    total_ids = [int(i) for i, l in id2label.items() if is_total_tag(to_tag(l))]
    p_total = probs[:, total_ids].sum(-1).cpu().numpy()

    gold = [to_tag(id2label[int(l)]) for l in labels[keep]]
    pred = [to_tag(id2label[int(p)]) for p in pred_ids[keep]]
    conf = [float(c) for c in confs[keep]]
    ptot = [float(x) for x in p_total[keep]]
    return gold, pred, conf, ptot

#Regex to look for total amount
AMOUNT_RE = re.compile(r"\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{2})?|\d+[.,]\d{2}")

def norm(s):
    s = re.sub(r"[^\d.\-]", "", s)
    if not any(ch.isdigit() for ch in s):
        return ""
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s

tok = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")


def words_from_example(ex):
    ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    starts = [i for i, l in enumerate(labels) if l != -100]
    ends = starts[1:] + [len(ids)]
    return [tok.decode(ids[a:b], skip_special_tokens=True).strip()
            for a, b in zip(starts, ends)]

def pick_total(words, p_total, min_prob=0.0):
    """Return the index of the single most likely TOTAL word, or None."""
    cands = [i for i, w in enumerate(words)
             if AMOUNT_RE.search(w) and p_total[i] >= min_prob]
    return max(cands, key=lambda i: p_total[i]) if cands else None

def eval_total(test_ds, all_gold, all_pred, all_ptotal, min_prob=0.0):
    n = n_skip = argmax_ok = pick_ok = pick_abstain = 0
    missed = wrong = extras = 0
    for ex, gold, pred, pt in zip(test_ds, all_gold, all_pred, all_ptotal):
        words = words_from_example(ex)
        g = {norm(w) for w, l in zip(words, gold) if is_total_tag(l)} - {""}
        if not g:
            n_skip += 1
            continue
        n += 1

        p_old = {norm(w) for w, l in zip(words, pred) if is_total_tag(l)} - {""}
        hit = bool(p_old & g)
        argmax_ok += hit
        if not p_old:                          missed  += 1
        elif not hit:                          wrong   += 1
        elif p_old - g:                        extras  += 1

        i = pick_total(words, pt, min_prob)
        if i is None:
            pick_abstain += 1
        else:
            pick_ok += norm(words[i]) in g

    print(f"[min_prob={min_prob}]  labeled={n}  skipped={n_skip}")
    print(f"  argmax   ok={argmax_ok}/{n} ({argmax_ok/n:.3f})")
    print(f"  pick-one ok={pick_ok}/{n} ({pick_ok/n:.3f})  "
          f"abstain={pick_abstain}  "
          f"missed={missed}  wrong={wrong}  extras={extras}")

def main():
        OUT_DIR.mkdir(exist_ok=True) #Creates a folder where we're going to save the results
        device = "cuda" if torch.cuda.is_available() else "cpu" #Checks if GPU is available for faster computation

        model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(device).eval() #defines the model
        _, _, test_ds = get_datasets() #retrieve the test set
        id2label = get_id2label() #retrieve the mapping from id to label

        all_gold, all_pred, all_conf, all_ptotal = [], [], [], []
        for ex in test_ds:
            g, p, c, pt = word_level_predictions(model, ex, id2label, device)
            all_gold.append(g); all_pred.append(p); all_conf.append(c); all_ptotal.append(pt)

        #--Evaluation--

        #classification report
        report_txt = classification_report(all_gold, all_pred, digits=4)
        print(report_txt)
        (OUT_DIR / "test_report.txt").write_text(report_txt, encoding="utf-8")

        for mp in SWEEP:
            
            eval_total(test_ds, all_gold, all_pred, all_ptotal, min_prob=mp)

        #Same report as JSON
        report_dict = classification_report(all_gold, all_pred, output_dict=True)
        ent = [c for gs, cs in zip(all_gold, all_conf) for g, c in zip(gs, cs) if g != "O"]
        err = [c for gs, ps, cs in zip(all_gold, all_pred, all_conf)
            for g, p, c in zip(gs, ps, cs) if g != p]
        summary = {
            "report": report_dict,
            "mean_conf_entity_words": sum(ent) / max(len(ent), 1),
            "mean_conf_on_errors": sum(err) / max(len(err), 1),
            "n_docs": len(all_gold),
        }
        (OUT_DIR / "test_metrics.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )

        #Per-document predictions for error analysis
        with open(OUT_DIR / "test_predictions.jsonl", "w", encoding="utf-8") as f:
            for i, (g, p, c) in enumerate(zip(all_gold, all_pred, all_conf)):
                f.write(json.dumps({"doc": i, "gold": g, "pred": p,
                                    "conf": [round(x, 4) for x in c]}) + "\n")


if __name__ == "__main__":
    main()