# 🧝 ElfWeave: Self-Improving Multi-Agent Ecosystem

ElfWeave is a high-performance, modular multi-agent platform designed for local execution. It is specifically optimized for hardware with **8GB VRAM (RTX 5060)** and 32GB RAM, utilizing a tiered LLM architecture to provide intelligent orchestration without crashing local resources.

## 🚀 Key Features

- **Standardized "Claude Code" UI**: A premium, terminal-native dashboard with **synchronized 10Hz animation frames** and unified result cards across all agents.
- **Fully Responsive Design**: Dynamic UI sizing that adapts fluidly to your terminal width using Rich-based rules and ratio-mapped tables.
- **Asynchronous Orchestration**: Fully `asyncio`-powered TUI with live token streaming and non-blocking sub-agent logs.
- **Specialist-First Routing**: Intelligent tool selection that prioritizes specialized agents (Weather, Browser, Knowledge) over raw utilities to prevent hallucinations.
- **Subprocess Robustness**: Integrated `stderr` capture and Python traceback visibility, ensuring that agent crashes are reported with full technical context.
- **Hardware-Aware Tuning**: Optimized for your local environment using **qwen2.5:7b** for planning and **llama3.1:8b** for validation and specialist assistance.

## 🛠 Project Structure

```text
src/
├── harness.py          # Central Orchestrator & Planner (qwen2.5:7b)
├── common/
│   ├── config.py       # Hardware & VRAM optimizations
│   ├── ollama.py       # Async LLM helpers & robust JSON parsing
│   └── ui.py           # Synchronized, Responsive TUI manager
└── modules/
    ├── browser_agent    # Autonomous web navigation
    ├── weather          # Multimodal weather analysis
    ├── knowledge_agent  # Local RAG & semantic search
    └── monitor_agent    # Real-time log/anomaly watcher
```

## 🧠 Specialist Priority Architecture

To minimize hallucination (e.g., LLMs inventing URLs for `http_get`), ElfWeave enforces a **Specialist Priority** protocol:
1. **Domain Specialists**: High-level modules like `weather` or `browser` are authoritative for their domains.
2. **General Utilities**: Tools like `http_get` or `shell` are only used when no specialist exists or when a specific URL/command is grounded in previous output.

## 📈 Validation & Self-Refinement

ElfWeave doesn't just fail; it analyzes and adapts.
1. **Validation Audit**: The Reviewer model (**llama3.1:8b**) scores every output. If it falls below 70%, a retry is triggered.
2. **Crash Visibility**: If an agent module fails (Exit Code 1), the harness captures the full **Python traceback** and presents it to the planner for autonomous debugging.
3. **Episodic Memory**: Past execution results are stored in `.harness_history.json` and injected into the Planner's context to avoid repeating past mistakes.

## 🤖 Agent Roster

| Agent | Domain | Model (Local) | Technology |
| :--- | :--- | :--- | :--- |
| **Browser** | Web Research | `qwen2.5:7b` | `browser-use`, `Playwright` |
| **Weather** | Meteorology | `llama3.1:8b` | `wttr.in`, `vision-llm` |
| **Knowledge** | Local RAG | `Local Embed` | `FAISS`, `MiniLM` |
| **Monitor** | Anomaly Detection | `llama3.1:8b` | `asyncio`, `Ollama` |

---
*Created by the Night-Traders-Dev team.*
