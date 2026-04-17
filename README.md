# 🧝 ElfWeave: Evolutionary Multi-Agent Ecosystem

ElfWeave is a high-performance, modular multi-agent platform designed for local execution. It is an **evolutionary ecosystem** where agents can not only perform complex tasks but also autonomously identify, analyze, and repair their own source code to recover from failures.

Specifically optimized for hardware with **8GB VRAM (RTX 5060)** and 32GB RAM, utilizing a tiered LLM architecture to provide intelligent orchestration without crashing local resources.

## 🚀 Key Features

- **Evolutionary Self-Repair**: Agents detect their own logic/syntax errors, call a self-analysis tool, and autonomously apply code patches using the `repair_code` engine.
- **Web-Learning Loop**: When an agent encounters an unfamiliar error, it proactively "Googles" the documentation via the `research_fix` specialist to learn the correct recovery strategy.
- **Persistent Experience Store**: Long-term memory at `~/.agent_experience.jsonl` tracks every success and failure, allowing the orchestrator to "remember" and avoid past mistakes.
- **Standardized "Claude Code" UI**: A premium, terminal-native dashboard with **synchronized 10Hz animation frames** and responsive, fluid layout cards.
- **Hardware-Aware Tuning**: Optimized for local environment using **qwen2.5:7b** for planning and **llama3.1:8b** for high-fidelity code generation and validation.

## 🛠 Project Structure

```text
src/
├── harness.py          # Central Orchestrator & Autonomous Self-Healer
├── common/
│   ├── config.py       # Hardware & VRAM optimizations
│   ├── ollama.py       # Async LLM helpers & robust JSON parsing
│   └── ui.py           # Synchronized, Responsive TUI manager
└── modules/
    ├── code_architect   # Design review & technical debt auditor (NEW)
    ├── fs_manager       # Repository tree & metadata manager (NEW)
    ├── browser_agent    # Autonomous web research specialist
    ├── weather          # Multimodal meteorology analyst
    ├── knowledge_agent  # Local RAG & semantic search
    └── monitor_agent    # Real-time resource & log observer
```

## 🧠 Autonomous Learning & Self-Audit

ElfWeave implements a "Closed-Loop" evolutionary lifecycle:
1. **Validation Audit**: The Reviewer model (**llama3.1:8b**) scores every task.
2. **Failure Analysis**: If a task fails, the `analyze_failure` tool performs a technical root-cause diagnosis.
3. **Research Fix**: For ambiguous errors, the agent performs a targeted web search to find best practices.
4. **Autonomous Repair**: The `repair_code` tool applies an LLM-generated patch to the module's source code on disk.
5. **Memory Retrieval**: The Planner consults past lessons from the Experience Store to avoid regressions.

## 🤖 Agent Roster & Examples

| Agent | Capability | Example Command |
| :--- | :--- | :--- |
| **Architect** | Design Review / Refactoring | `uv run ... "Perform an architectural audit of the project structure"` |
| **FS Manager** | Project Visualization | `uv run ... "Show me the project structure in the modules directory"` |
| **Browser** | Deep Web Research | `uv run ... "Find the latest release notes for the browser-use library"` |
| **Weather** | Multimodal Forecasting | `uv run ... "Weather for Ashland Kentucky"` |
| **Knowledge** | Semantic Repo Search | `uv run ... "Search the knowledge base for LLM context config"` |
| **Monitor** | Resource Observation | `uv run ... "Monitor GPU and VRAM usage for anomalies"` |
| **Self-Healer** | Evolutionary Evolution | `uv run ... "Identify and fix the NameError in weather.py"` |

## 📦 Getting Started

1. **Install Dependencies**: `uv sync`
2. **Setup Ollama**: Ensure `ollama` is running with `llama3.1:8b` and `qwen2.5:7b` pulled.
3. **Run a Mission**:
   ```bash
   uv run --with browser-use --with ollama python src/harness.py "Show me the project structure"
   ```

---
*Created by the Night-Traders-Dev team.*
