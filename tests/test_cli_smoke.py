import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class CliSmokeTests(unittest.TestCase):
    def test_harness_requires_query_in_noninteractive_mode(self) -> None:
        proc = run_cli("src/harness.py")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Error: no query provided.", proc.stderr)

    def test_main_lists_tools(self) -> None:
        proc = run_cli("main.py", "--list-tools")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("knowledge_query", proc.stdout)

    def test_fs_manager_runs(self) -> None:
        proc = run_cli("src/modules/fs_manager.py", ".", "--harness")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Repository Structure", proc.stdout)

    def test_knowledge_agent_query_runs(self) -> None:
        proc = run_cli("src/modules/knowledge_agent.py", "--query", "harness", "--harness")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("File:", proc.stdout)

    def test_browser_agent_help_runs(self) -> None:
        proc = run_cli("src/modules/browser_agent.py", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Autonomous Browser Agent", proc.stdout)

    def test_monitor_agent_help_runs(self) -> None:
        proc = run_cli("src/modules/monitor_agent.py", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Real-time Log Monitor Agent", proc.stdout)


if __name__ == "__main__":
    unittest.main()
