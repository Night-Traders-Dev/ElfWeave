import asyncio
from typing import Optional, Any, Callable

# ══════════════════════════════════════════════════════════════════════
#  Browser Logic
# ══════════════════════════════════════════════════════════════════════

class BrowserAgentLogic:
    def __init__(self, model: str, host: str, headless: bool = True):
        try:
            from browser_use import Agent, Browser, BrowserConfig
            from langchain_ollama import ChatOllama
        except ModuleNotFoundError as exc:
            missing = exc.name or "browser agent dependency"
            raise RuntimeError(
                f"Missing optional browser dependency: {missing}. "
                "Install the browser extras or run via `uv run --with browser-use --with langchain-ollama`."
            ) from exc

        self.model = model
        self.host = host
        self.headless = headless
        self._agent_cls = Agent
        
        # Configure the LLM via LangChain (required by browser-use)
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.host,
            num_ctx=32000, # Large context needed for DOM snapshots
            temperature=0.0
        )
        
        self.browser = Browser(
            config=BrowserConfig(
                headless=self.headless,
                disable_security=True, # Often needed for local scraping
            )
        )

    async def run_task(self, task: str, ui_callback: Optional[Callable[[str, str], None]] = None) -> str:
        """
        Runs a browser task and optionally reports progress via ui_callback(step_name, detail).
        """
        agent = self._agent_cls(
            task=task,
            llm=self.llm,
            browser=self.browser,
        )
        
        # NOTE: browser-use doesn't have a simple per-step callback that matches our Step UI perfectly
        # without wrapping the internal controller, so for now we report the start and end.
        if ui_callback:
            ui_callback("initializing browser", "starting playwright...")
            
        try:
            result = await agent.run()
            # The result is an AgentHistory object, we want the final outcome
            return str(result.final_result() or "Task completed with no explicit return value.")
        except Exception as e:
            return f"Browser Error: {str(e)}"
        finally:
            await self.browser.close()

async def execute_browser_task(task: str, model: str, host: str, ui: Any, refresh: Callable[[], None], harness: bool = False) -> str:
    logic = BrowserAgentLogic(model=model, host=host, headless=harness)
    
    s_browser = ui.add_step("browser session").start(); refresh()
    
    def ui_cb(name: str, detail: str):
        # We can update the same step or add new ones
        s_browser.detail = f"{name}: {detail}"
        refresh()

    res = await logic.run_task(task, ui_callback=ui_cb)
    
    s_browser.done(res[:60]); refresh()
    return res
