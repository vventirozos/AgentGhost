words from the human:
This agent performs very well with qwen 3.5, and it's specially alligned for Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf which (from my tests) proved to be the best uncensored version of the model. the current setup i have is :
1. a mac mini m4 running llama-server and the agent.
2. a Jetson nano running Stable diffusion 1.5 for image generation (it's pretty good given it's a 2GB model.)
3. a raspberry pi hosting TTS and STT (it's all local and it's not that good)

The agent has a lot of intricacies and tricks, for example if you ask it to "generate a picture of a dog", and you don't like it you can say i "don't like it, use your imagination to create a new one." The agent will read the picture using the visual part of qwen and generate a new prompt for stable diffusion. 
Something that the AI forgets to mention in the documentation bellow is that there's a web client included, be sure to check it out, it's pretty cool.

The rest is AI generated:


# Ghost Agent 👻

Ghost Agent is an autonomous, scalable, and highly capable AI service designed to operate as a proactive agent. Built on top of `fastapi` and utilizing large language models (LLMs), it goes beyond simple chat completions—it continuously plans, learns, and builds on its experiences through advanced memory systems, tool execution sandboxes, and biological-inspired behavioral hooks.

## 🚀 Key Features

*   **Autonomy & Biological Hooks**: Ghost Agent isn't just reactive. It watches for idle periods and can enter spontaneous "REM Dreaming" to consolidate its memory. Background watchdog mechanisms ensure the agent periodically processes its short-term memory journal and reflects on past interactions.
*   **Swarm & Multi-Node Architecture**: Offload workloads by distributing them across specialized worker nodes!
    *   `--swarm-nodes`: Hand off reasoning to multiple AI endpoints.
    *   `--worker-nodes`: Keep edge and background tasks independent.
    *   `--visual-nodes` / `--image-gen-nodes`: Delegate vision analytics and image generation.
    *   `--coding-nodes`: Leverage specialized coding models.
*   **Advanced Memory Subsystems**: Ghost Agent boasts a complex, multi-tiered memory architecture implemented with ChromaDB Vector representations:
    *   **Vector Memory**: Semantic, long-term embeddings of facts and history.
    *   **Profile Memory**: Implicit details about the user and preferences.
    *   **Skill Memory**: Persistent learning and mastery of new tool paradigms.
    *   **Journal**: Short-term logs of real-time executions that are processed during idle time.
    *   **Smart Selectivity**: Adaptive thresholding (`--smart-memory`) allows the agent to smartly curate what it decides to commit to its permanent brain.
*   **Secure Docker Sandbox Execution**: The agent safely runs complex operations or terminal commands without tearing down your host OS. The sandbox handles everything from isolating the execution environment to proxying connections through Tor (if configured).
*   **Anonymous Search By Default**: For enhanced privacy, Ghost Agent utilizes Tor combined with DuckDuckGo for strictly anonymous web searches.
*   **Multi-Interface Support**:
    *   **FastAPI REST Web Server**: Exposes standard endpoints (`/api/chat`, `/api/upload`, `/api/download`) alongside streaming outputs.
    *   **Slack Bot Integration**: A completely featured async Slack Bot that hooks into Slack's Socket Mode. It maintains full thread contexts, downloads user files to the secure sandbox automatically, uploads agent-generated images straight to the conversation, and provides live emoji-based status updates (e.g., 🧠 Thinking..., 🔬 Researching..., 🎨 Generating Image...) as it streams logs!
    *   **Audio Proxing**: Supports interfacing with remote Audio endpoints (e.g., Raspberry Pi) for Speech-to-Text and Text-to-Speech interactions.

---

## 🏗️ Architecture

Ghost Agent separates concerns heavily between logic extraction and execution models:

1.  **Core Agent (`src/ghost_agent/core/`)**: Houses the primary intelligence. Here you'll find logic to handle chat payloads, interface with vector databases, manage planning phases, and initiate dreams (`dream.py`) or self-reflection schemas.
2.  **Memory Management (`src/ghost_agent/memory/`)**: The bridge to persistent storage (ChromaDB + SQLite). It splits incoming contexts into semantic chunks and maintains a chronological graph of short-term tasks.
3.  **Tools & Registry (`src/ghost_agent/tools/`)**: Ghost Agent equips itself dynamically. The registry maps out everything from `file_system` utilities and `execute` operations, to generic `search`, `tasks`, and standard `system` commands.
4.  **Sandbox (`src/ghost_agent/sandbox/`)**: Uses the `docker` API to spin up isolated container instances (defaulting to alpine or python derivatives) to run untrusted code blocks derived from the LLM.
5.  **Interface layer (`interface/`)**: The presentation boundary. Contains the `server.py` implementation to pipe fast websockets and log streaming, plus the distinct `slack_bot` logic for enterprise or team integration.

---

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/AgentGhost.git
    cd AgentGhost
    ```

2.  **Set up a Virtual Environment and Install Dependencies**
    Ghost Agent uses modern Python > 3.10 and relies on multiple heavy ML libraries (`transformers`, `sentence-transformers`, `torch`), as well as API servers (`fastapi`, `uvicorn`).
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Ensure Docker is Running**
    The code execution abilities explicitly hinge on a local Docker daemon being active. If Docker is offline, the sandbox initialization will display a warning and code execution tools may fail.

4.  **Set Environment Variables**
    Ghost Agent utilizes multiple environmental variables to configure its behavior:
    ```bash
    export GHOST_API_KEY="your-secret-key-limitless-access"
    export GHOST_HOME="/path/to/ghost_llamacpp"
    export GHOST_MODEL="qwen-3.5-9b"  # Defaults to qwen
    export TOR_PROXY="socks5://127.0.0.1:9050" # Proxy for anonymous interactions
    export GHOST_SANDBOX_DIR="/tmp/sandbox"

    # Slack Support
    export SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"
    export SLACK_APP_TOKEN="xapp-your-slack-app-token"
    ```

---

## ⌨️ Usage

### Starting the Core Server

The core logic must be running and listening for the interfaces to succeed. Use `python -m src.ghost_agent.main` module execution:

```bash
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --host "0.0.0.0" \
    --port 8000 \
    --verbose
```

**Key Command-Line Arguments:**
*   `--upstream-url`: The endpoint string hosting your local or remote LLM compatible API (Ollama, vLLM, etc).
*   `--swarm-nodes` / `--visual-nodes` / `--worker-nodes`: Comma-separated list of `url|model` to utilize a decentralized inference topology.
*   `--smart-memory`: Set a value > 0 (e.g., `0.5`) to turn on advanced curation of semantic memories.
*   `--no-memory`: Starts the agent with amnesia (mostly used for debugging/regression testing).
*   `--perfect-it`: A special flag to turn on proactive code optimization feedback loops after a session.

### Running the FastAPI Interface

The API and visual web proxy are available under `interface/server.py`.
```bash
python interface/server.py --agent-log /Users/vasilis/AI/Logs/ghost-agent.log
```

### Running the Slack Bot

To connect Ghost to Slack, boot up the socket bot. Be sure you have exported `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`.
```bash
python interface/slack_bot/main.py --log-file /Users/vasilis/AI/Logs/ghost-slack-bot-main.err
```

### Interacting

*   **Browser**: Go to `http://localhost:8080` to access the chat and streaming log UI.
*   **Slack**: `@Ghost` mention the bot in channels or send it a direct message! Ghost will spawn threads automatically to handle context, display live thought processes using emojis, ingest PDFs or files you send to it, and output markdown formatted texts and system-generated images.

---

## 🛡️ Telemetry Notice

Ghost Agent aggressively disables tracking natively by hooking into standard environment variables. PostHog, telemetry from ChromaDB, and HF Hub hooks are **disabled by default**, ensuring that your sandbox and memories remain entirely yours.

