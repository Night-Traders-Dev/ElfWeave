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
- **Optional Megakernel Backend**: ElfWeave can experimentally route selected planning and validation phases through the Luce Megakernel backend while keeping Ollama as the safe default.
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

## ⚙ Optional Megakernel Backend

ElfWeave now has an **experimental** backend adapter for [Luce Megakernel](https://github.com/Luce-Org/luce-megakernel). This is optional and disabled by default.

Current behavior:

- Ollama remains the default backend.
- Megakernel can be enabled for selected phases such as `planner`, `sanity`, and `validator`.
- If the Megakernel path is unavailable or fails, ElfWeave falls back to Ollama by default.
- This path is currently best suited for **small text-only orchestration phases**, not the full platform.
- The repository now vendors Luce Megakernel as the `third_party/luce-megakernel` submodule.

Clone with submodules, or initialize them after cloning:

```bash
git submodule update --init --recursive
```

One-time kernel build on the `elf_g` device:

```bash
cd third_party/luce-megakernel
pip install -e .
pip install transformers accelerate
```

Then run ElfWeave with the bundled hybrid defaults:

```bash
uv run --with browser-use --with ollama python main.py --use-kernel "weather ashland kentucky"
```

`--use-kernel` applies the same default behavior as:

```bash
export ELFWEAVE_INFERENCE_BACKEND=hybrid
export ELFWEAVE_MEGAKERNEL_REPO=/home/kraken/elf_g/ElfWeave/third_party/luce-megakernel
export ELFWEAVE_MEGAKERNEL_MODEL=Qwen/Qwen3.5-0.8B
export ELFWEAVE_MEGAKERNEL_PHASES=planner,sanity,validator,analyzer,summarizer
```

Those values are still overridable with explicit environment variables.

Available environment flags:

- `ELFWEAVE_INFERENCE_BACKEND=ollama|hybrid|megakernel`
- `ELFWEAVE_MEGAKERNEL_REPO=/path/to/luce-megakernel`
- `ELFWEAVE_MEGAKERNEL_MODEL=Qwen/Qwen3.5-0.8B`
- `ELFWEAVE_MEGAKERNEL_PHASES=planner,sanity,validator,...`
- `ELFWEAVE_MEGAKERNEL_MAX_TOKENS=256`
- `ELFWEAVE_MEGAKERNEL_FALLBACK=1`
- `ELFWEAVE_MEGAKERNEL_CUDA_ARCH=86` to force a specific CUDA arch when building the submodule

What changed for ElfWeave compatibility:

- ElfWeave now prefers the vendored submodule path automatically instead of requiring a separate clone.
- The vendored megakernel build now targets the active GPU architecture or `ELFWEAVE_MEGAKERNEL_CUDA_ARCH`, instead of hard-coding `sm_86`.
- ElfWeave only routes **short text-only** phases to Luce Megakernel and keeps long prompts on Ollama, because the upstream kernel currently exposes a 2048-token context window in `third_party/luce-megakernel/model.py`.

Important limitations:

- Luce Megakernel is architecture-specific and currently targets **Qwen 3.5-0.8B hybrid DeltaNet/Attention**, not ElfWeave's default Ollama models.
- This integration is an adapter layer, not a drop-in replacement for Ollama.
- Vision/image requests and unsupported phases stay on Ollama.
- If the compiled CUDA extension or `torch`/`transformers` dependencies are missing, `--use-kernel` falls back to Ollama and keeps the mission running.

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
