# 🧝 ElfWeave: Self-Improving Multi-Agent Ecosystem

ElfWeave is a high-performance, modular multi-agent platform designed for local execution. It is specifically optimized for hardware with **8GB VRAM (RTX 5060)** and 32GB RAM, utilizing a tiered LLM architecture to provide intelligent orchestration without crashing local resources.

## 🚀 Key Features

- **Asynchronous Orchestration**: Fully `asyncio`-powered TUI with live token streaming and non-blocking sub-agent logs.
- **Self-Improvement Loop**: Autonomous retry mechanism with LLM-based failure analysis and episodic memory.
- **Hardware-Aware Tuning**: Tiered model approach (**Planner** for reasoning vs. **Reviewer** for validation) with dynamic context window management.
- **Domain Knowledge Injection**: Agents "train" themselves on-the-fly by querying specialized markdown manuals before execution.
- **Rich TUI**: A beautiful, terminal-native dashboard featuring fluid animations and real-time status tracking.

## 🛠 Project Structure

```text
src/
├── harness.py          # Central Orchestrator & Planner
├── common/
│   ├── config.py       # Hardware & VRAM optimizations
│   ├── ollama.py       # Async LLM helpers & streaming
│   └── ui.py           # Rich TUI state management
└── modules/
    ├── browser_agent    # Autonomous web navigation
    ├── weather          # Multimodal weather analysis
    ├── knowledge_agent  # Local RAG & semantic search
    └── monitor_agent    # Real-time log/anomaly watcher
```

## 🚥 Getting Started

### Prerequisites
- [Ollama](https://ollama.ai/) installed and running.
- [uv](https://github.com/astral-sh/uv) for fast, isolated python execution.

### Installation
Clone the repository and run the harness. `uv` will automatically handle dependencies:

```bash
uv run --with browser-use --with ollama python src/harness.py "Get the weather for Ashland, KY"
```

## 🧠 Tiered Model Architecture

To thrive on an **8GB RTX 5060**, ElfWeave splits duties between models:
- **Planner (qwen2.5:7b)**: Handles complex reasoning, tool selection, and task execution.
- **Reviewer (llama3.2:3b)**: Validates outputs, checks sanity, and audits grounding.
- **Monitor (llama3.2:1b)**: Optimized for high-frequency log analysis and anomaly detection.

## 📈 Self-Improvement & Memory

ElfWeave doesn't just fail; it learns.
1. **Validation Audit**: The Reviewer model scores every output. If it falls below 70%, a retry is triggered.
2. **Failure Analysis**: The `analyze_failure` tool examines the orchestrator code and logs to suggest a specific fix for the next retry.
3. **Episodic Memory**: Past execution results are stored in `.harness_history.json` and injected into the Planner's context to avoid repeating past mistakes.

## 🤖 Agent Roster

| Agent | Task | Tech Stack |
| :--- | :--- | :--- |
| **Browser** | Autonomous web research | `browser-use`, `Playwright` |
| **Weather** | Real-time multimodal weather | `wttr.in`, `vision-llm` |
| **Knowledge** | Local codebase RAG | `FAISS`, `SentenceTransformers` |
| **Monitor** | Real-time anomaly detection | `async generators`, `LogTailer` |

---
*Created by the Night-Traders-Dev team.*
