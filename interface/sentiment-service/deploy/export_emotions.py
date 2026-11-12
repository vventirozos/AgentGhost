"""One-time export of the EMOTION model to ONNX (§4UH-warmth), run in the same throwaway venv as export_model.py.

SamLowe/roberta-base-go_emotions: 28 labels (27 emotions + neutral) trained on GoEmotions (Reddit comments),
MULTI-label: each label is its own sigmoid, so several can be true at once. Writes model.onnx, the tokenizer,
labels.json and parity.json (the original model's sigmoid probabilities, checked by the service at start).
usage: python export_emotions.py <out dir>"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "SamLowe/roberta-base-go_emotions"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
tok.save_pretrained(out)
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
(out / "labels.json").write_text(json.dumps(labels))
enc = tok(["this is great", "a much longer sentence to vary the length a little"], padding=True, return_tensors="pt")
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(out / "model.onnx"),
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
                  opset_version=17, dynamo=False)
samples = ["thanks, that worked perfectly", "this is wrong again, you broke it", "list the files in the sandbox",
           "haha that's hilarious"]
with torch.no_grad():
    ref = [torch.sigmoid(model(**tok(s, return_tensors="pt")).logits)[0].tolist() for s in samples]
(out / "parity.json").write_text(json.dumps({"samples": samples, "probs": ref, "activation": "sigmoid"}))
for s, r in zip(samples, ref):
    top = sorted(zip(labels, r), key=lambda x: -x[1])[:3]
    print(s, [(lab, round(p, 3)) for lab, p in top])
