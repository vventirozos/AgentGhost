"""One-time export of the sentiment model to ONNX, run in a throwaway venv with torch + transformers (see README).
Writes model.onnx, the tokenizer, labels.json and parity.json (the original model's answers, checked by the service at start)."""
import sys, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
repo = "cardiffnlp/twitter-roberta-base-sentiment-latest"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
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
# reference logits for the parity check
samples = ["thanks, that worked perfectly", "this is wrong again, you broke it", "list the files in the sandbox"]
with torch.no_grad():
    e = tok(samples, padding=True, return_tensors="pt")
    ref = torch.softmax(model(**e).logits, -1).tolist()
(out / "parity.json").write_text(json.dumps({"samples": samples, "probs": ref}))
print(labels, [[round(x, 3) for x in r] for r in ref])
