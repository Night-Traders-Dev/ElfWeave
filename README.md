# 🧝 ElfWeave: Self-Improving Multi-Agent Ecosystem

ElfWeave is a high-performance, modular multi-agent platform designed for local execution. It is specifically optimized for hardware with **8GB VRAM (RTX 5060)** and 32GB RAM, utilizing a tiered LLM architecture to provide intelligent orchestration without crashing local resources.

## 🚀 Key Features

- **Standardized "Claude Code" UI**: A premium, terminal-native dashboard with **synchronized 10Hz animation frames** and unified result cards across all agents.
- **Asynchronous Orchestration**: Fully `asyncio`-powered TUI with live token streaming and non-blocking sub-agent logs.
- **Specialist-First Routing**: Intelligent tool selection that prioritizes high-level agents (Weather, Browser, Knowledge) over raw utilities to prevent hallucinations.
- **Self-Improvement Loop**: Autonomous retry mechanism with LLM-based failure analysis and episodic memory.
- **Hardware-Aware Tuning**: Tiered model approach (**qwen2.5:7b** for reasoning vs. **llama3.2:3b** for validation) with dynamic context window management.

## 🛠 Project Structure

```text
src/
├── harness.py          # Central Orchestrator & Planner (qwen2.5:7b)
├── common/
│   ├── config.py       # Hardware & VRAM optimizations
│   ├── ollama.py       # Async LLM helpers & streaming
│   └── ui.py           # Synchronized TUI lifecycle manager
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

## 📈 Self-Improvement & Memory

ElfWeave doesn't just fail; it learns.
1. **Validation Audit**: The Reviewer model (`llama3.2:3b`) scores every output. If it falls below 70%, a retry is triggered.
2. **Failure Analysis**: The `analyze_failure` tool examines the orchestrator code and logs to suggest a specific fix for the next retry.
3. **Episodic Memory**: Past execution results are stored in `.harness_history.json` and injected into the Planner's context to avoid repeating past mistakes.

## 🤖 Agent Roster

| Agent | Domain | Technology |
| :--- | :--- | :--- |
| **Browser** | Web Research | `browser-use`, `Playwright` |
| **Weather** | Meteorology | `wttr.in`, `vision-llm` |
| **Knowledge** | Local RAG | `FAISS`, `MiniLM` |
| **Monitor** | Anomaly Detection | `asyncio`, `Ollama` |

---
*Created by the Night-Traders-Dev team.*
