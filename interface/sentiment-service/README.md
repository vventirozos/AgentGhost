# sentiment-service

The sentiment of the owner's English messages, for the agent's mood (§4TW). It uses [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) (negative / neutral / positive, 125M parameters), exported once to ONNX and run with onnxruntime on the worker node's CPU. A short message takes a few milliseconds there, and a 200-token one takes about 160 ms. The process holds about 870 MB.

Since 2026-11-12 (§4UH-warmth) it can also read the message's EMOTIONS with a second model, [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) (28 labels, multi-label), loaded from `~/Data/AI/sentiment-service/emotions/` when that directory has a `model.onnx` (about 480 MB; the process then holds about 1.25 GB). The character's warmth dial reads them; the mood still reads only the three-way sentiment.

The agent calls it from `selfhood/sentiment.py`, in the background, for owner messages only. A confident positive or negative score (0.85 or more) nudges the mood's pleasantness by ±0.08, and every score is journalled in `sentiment.jsonl` beside the mood journal. See `docs/algorithms/selfhood.html#sentiment`.

The model reads English only, and the service checks that before classifying. A message whose letters are less than 70 % Latin (Greek letters) answers `unknown`, and so does one with fewer than two letters. Then fastText's language identifier [lid.176](https://fasttext.cc/docs/en/language-identification.html) must name English first at 0.4 or more: Greek typed in Latin letters fails here (`not English`), and so does a word it does not know, which gets a flat prior of 0.12 (`language unclear`).

## Files on the service machine

| Path | What it holds |
|---|---|
| `~/Data/AI/sentiment-service/app/sentiment_service.py` | This code |
| `~/Data/AI/sentiment-service/start-sentiment-service.sh` | launchd entry point: binds the tailnet address |
| `~/Data/AI/sentiment-service/venv/` | Python 3.10 with `onnxruntime`, `tokenizers`, `numpy`, `fasttext-wheel` |
| `~/Data/AI/sentiment-service/langid/lid.176.ftz` | fastText language identifier (938 KB) |
| `~/Data/AI/sentiment-service/model/` | `model.onnx`, `tokenizer.json`, `labels.json`, `parity.json` |
| `~/Data/AI/sentiment-service/emotions/` | The emotion model: the same four files; `parity.json` says `"activation": "sigmoid"` |
| `~/Data/AI/.sentiment_service_key` | Shared key, 0600, also on the agent host |
| `~/Data/AI/Logs/sentiment-service.log` | Routes, labels and exception types, never message text |

## Set up

onnxruntime publishes no wheels for Python 3.15, so the venv uses 3.10.

```
S=~/Data/AI/sentiment-service; mkdir -p "$S/app"
python3.10 -m venv "$S/venv" && "$S/venv/bin/pip" install onnxruntime tokenizers numpy fasttext-wheel
mkdir -p "$S/langid" && curl -sSL -o "$S/langid/lid.176.ftz" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

fasttext-wheel's `predict` fails under numpy 2, so the service calls the extension's own predict.

The model repository has no ONNX file, so export it once in a throwaway venv, the only step that needs the internet:

```
python3.10 -m venv /tmp/sent-export && /tmp/sent-export/bin/pip install torch transformers onnx onnxscript
/tmp/sent-export/bin/python deploy/export_model.py "$S/model"
rm -rf /tmp/sent-export
```

`deploy/export_model.py` loads the model with `AutoModelForSequenceClassification`, saves the tokenizer and `labels.json` (the `id2label` order), runs `torch.onnx.export` with inputs `input_ids` and `attention_mask`, output `logits`, dynamic batch and sequence axes, opset 17, and writes `parity.json`: three sample texts and the original model's probabilities for them. The service checks the ONNX model against those at start and refuses to run if any probability differs by more than 0.01. An int8 quantized copy was 3 times faster and shifted probabilities by up to 0.06, so the fp32 export is the one in use.

The emotion model is exported the same way, in the same throwaway venv: `/tmp/sent-export/bin/python deploy/export_emotions.py "$S/emotions"`. The service loads it if the directory is there (`--emotions-dir`), and checks its sigmoid parity file at start.

Copy `sentiment_service.py` to `app/` and `deploy/start-sentiment-service.sh` to `$S/`. Fill `__USER__` and `__HOME__` in `deploy/com.local.sentiment-service.plist`, install it to `/Library/LaunchDaemons/` owned by root, then run `sudo launchctl bootstrap system /Library/LaunchDaemons/com.local.sentiment-service.plist`.

On the agent host, put the same key in `~/Data/AI/.sentiment_service_key` and start the agent with `--sentiment-url http://<service tailnet address>:8097` (or export `GHOST_SENTIMENT_URL`; the flag wins).

## Run it

```
sudo launchctl print system/com.local.sentiment-service
sudo launchctl kickstart -k system/com.local.sentiment-service
```

## API

- `GET /health` answers `{"ok": true}`, without a key. The model is loaded and has passed its parity check before the port opens.
- `POST /classify` with header `X-Sentiment-Service-Key` and body `{"text": "..."}` answers `{"ok": true, "label": "positive", "score": 0.95, "probs": {"negative": 0.01, "neutral": 0.04, "positive": 0.95}, "reason": ""}`, plus `"emotions": {"gratitude": 0.99, ...}` (each label at 0.05 or more) when the emotion model is loaded. `label` is `unknown`, with `probs` empty, when the text is not English (`reason: "not English"`), when the identifier is unsure (`"language unclear"`) or when it has no words (`"no words"`). Text is limited to 4000 characters, and the model reads the first 512 tokens.
