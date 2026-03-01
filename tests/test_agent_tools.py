import unittest
from pathlib import Path

from src.core.agent import AgentCoworker, RepoSearchTool, ToolCall, ToolRegistry


class AgentToolTests(unittest.TestCase):
    def test_tool_call_json_schema(self):
        call = ToolCall(tool_name="repo_search", args={"query": "CodeMind"})
        payload = call.to_json()
        self.assertIn('"tool_name": "repo_search"', payload)
        self.assertIn('"args"', payload)

    def test_agent_runs_registered_tool(self):
        registry = ToolRegistry()
        registry.register(RepoSearchTool(Path(".").resolve()))
        agent = AgentCoworker(registry=registry, max_retries=1)
        result = agent.run("search CodeMind")
        self.assertIn("trace", result)
        self.assertGreaterEqual(len(result["trace"]), 1)


if __name__ == "__main__":
    unittest.main()
