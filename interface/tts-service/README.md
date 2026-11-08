# tts-service

English speech for spoken replies, using [Phonon 31M](https://huggingface.co/gradium/phonon-31m-en) (Apache 2.0) on the worker node's CPU. It takes about 0.03 s to make 1 s of audio there, and the weights file is 65 MB.

`interface/voice.py` calls it from `/api/tts`, so the web client and the handheld both use it. Greek stays on the macOS voice `GHOST_TTS_VOICE_EL`, because this model is English only. If the service fails, the sentence is spoken by `say` with `GHOST_TTS_VOICE`, and the interface doesn't try the service again for 30 s (`GHOST_TTS_RETRY_S`).

## Files on the service machine

| Path | What it holds |
|---|---|
| `~/Data/AI/tts-service/app/tts_service.py` | This code |
| `~/Data/AI/tts-service/start-tts-service.sh` | launchd entry point: binds the tailnet address, offline |
| `~/Data/AI/tts-service/venv/` | Python with `ptts` |
| `~/Data/AI/tts-service/hf/` | Weights and voice files |
| `~/Data/AI/.tts_service_key` | Shared key, 0600, also on the interface host |
| `~/Data/AI/Logs/tts-service.log` | Routes and exception types, never reply text |

## Set up

```
S=~/Data/AI/tts-service; mkdir -p "$S/app"
python3 -m venv "$S/venv" && "$S/venv/bin/pip" install ptts
HF_HOME="$S/hf" "$S/venv/bin/python" -c "import ptts; ptts.TTS('hf://gradium/phonon-31m-en', lang='en', quant='q8', device='cpu').save('/tmp/check.wav', 'Check.', voice='Lottie')"
```

That last line is the only step that needs the internet: it caches the weights and the voice. Each voice is cached the first time it is used, so run it again with `HF_HOME` set before switching to a new one. The service itself runs offline.

Copy `tts_service.py` to `app/` and `deploy/start-tts-service.sh` to `$S/`. Fill `__USER__` and `__HOME__` in `deploy/com.local.tts-service.plist`, install it to `/Library/LaunchDaemons/` owned by root, then run `sudo launchctl bootstrap system /Library/LaunchDaemons/com.local.tts-service.plist`.

On the interface host, put the same key in `~/Data/AI/.tts_service_key` and export `GHOST_TTS_URL=http://<service tailnet address>:8096` in the client launcher.

## Run it

```
sudo launchctl print system/com.local.tts-service
sudo launchctl kickstart -k system/com.local.tts-service
```

To change the voice, add `--voice <name>` to the start script after caching that voice as above, then use the `kickstart` line. `ptts.catalog_voices()` lists them; the catalogue has 37 English voices.

## API

- `GET /health` answers `{"ok": true}`, without a key. The model is loaded and has spoken a test word before the port opens.
- `POST /speak` with header `X-Tts-Service-Key` and body `{"text": "..."}` answers 16-bit mono 24 kHz WAV. Text is limited to 4000 characters.
