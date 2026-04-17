# 🧝 ElfWeave: Evolutionary Multi-Agent Ecosystem

ElfWeave is a high-performance, modular multi-agent platform designed for local execution. It is an **evolutionary ecosystem** where agents can not only perform complex tasks but also autonomously identify, analyze, and repair their own source code to recover from failures.

Specifically optimized for hardware with **8GB VRAM (RTX 5060)** and 32GB RAM, utilizing a tiered LLM architecture to provide intelligent orchestration without crashing local resources.

## 🚀 Key Features

- **Evolutionary Self-Repair**: Agents detect their own logic/syntax errors, call a self-analysis tool, and autonomously apply code patches using the `repair_code` engine.
- **Web-Learning Loop**: When an agent encounters an unfamiliar error, it proactively "Googles" the documentation via the `research_fix` specialist to learn the correct recovery strategy.
- **Persistent Experience Store**: Long-term memory at `~/.agent_experience.jsonl` tracks every success and failure, allowing the orchestrator to "remember" and avoid past mistakes.
- **Responsive Terminal UI**: Shared live status, cards, and summaries now adapt to narrow and wide terminals so agents stay readable from compact shells to full-screen panes.
- **Consistent Harness Output**: Specialist agents emit concise harness-mode summaries so the orchestrator shows one clean result card instead of nested dashboards.
- **Lower Tooling Overhead**: Specialist subprocesses now reuse the active Python interpreter instead of spawning nested `uv run` environments inside a mission.
- **Relevance-Based Memory**: The planner now retrieves the most relevant prior successes and failures for the current query instead of only the most recent history.
- **Hardware-Aware Tuning**: Optimized for local environment using **qwen2.5:7b** for planning and **llama3.1:8b** for high-fidelity code generation and validation.

## 🛠 Project Structure

```text
src/
├── harness.py               # Slim Entry Point
├── harness_logic.py         # Execution engine & Tool Registry
├── harness_planner.py       # LLM Coordination & Prompts
├── harness_ui.py            # Orchestrator Rendering
├── common/
│   ├── config.py            # Global Hardware & Prompt Config
│   ├── ollama.py            # Async LLM Helpers
│   ├── types.py             # Shared Data Models
│   └── ui.py                # Synchronized TUI Manager
└── modules/
    ├── weather.py           # Entry: Weather Specialist
    ├── browser_agent.py     # Entry: Web Research Specialist
    ├── code_architect.py    # Entry: Design Auditor
    ├── code_architect_logic.py 
    ├── code_architect_ui.py
    ├── fs_manager.py        # Entry: Repository Explorer
    ├── fs_manager_logic.py
    └── fs_manager_ui.py
```

## 🧠 Autonomous Learning & Self-Audit

ElfWeave implements a "Closed-Loop" evolutionary lifecycle:
1. **Validation Audit**: The Reviewer model (**llama3.1:8b**) scores every task.
2. **Failure Analysis**: If a task fails, the `analyze_failure` tool performs a technical root-cause diagnosis.
3. **Research Fix**: For ambiguous errors, the agent performs a targeted web search to find best practices.
4. **Autonomous Repair**: The `repair_code` tool applies an LLM-generated patch to the module's source code on disk.
5. **Memory Retrieval**: The Planner consults past lessons from the Experience Store to avoid regressions.
6. **Relevant Retry Feedback**: Failed runs feed tool output and validator guidance back into the next planning attempt so the system can actively learn inside a single mission.

## 🖥 UI Behavior

- **Full mode**: standalone agents render richer tables and panels when there is enough terminal width.
- **Compact mode**: tables shed lower-value columns before they overflow.
- **Harness mode**: specialists return plain, width-conscious summaries so the top-level harness owns the final presentation.
- **Shared chrome**: step rows, cards, and validation blocks all use the same responsive sizing rules from `src/common/ui.py`.
- **Interpreter reuse**: tool subprocesses inherit the live mission environment, which removes nested `uv` environment churn and reduces startup overhead.

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
   uv run --with browser-use --with ollama python main.py "Show me the project structure"
   ```

## ✅ Verification

Run the smoke and responsive-layout checks with:

```bash
python -m unittest discover -s tests -v
```

---
*Created by the Night-Traders-Dev team.*
