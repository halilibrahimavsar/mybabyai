from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"tool_name": self.tool_name, "args": self.args}, ensure_ascii=False)


@dataclass
class ToolResult:
    tool_name: str
    args: Dict[str, Any]
    ok: bool
    result: str
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AgentTool(Protocol):
    name: str
    description: str

    def run(self, args: Dict[str, Any]) -> ToolResult:
        ...


class RepoSearchTool:
    name = "repo_search"
    description = "Search text in repository using ripgrep."

    def __init__(self, root: Path):
        self.root = root

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(self.name, args, False, "", "missing query")
        try:
            proc = subprocess.run(
                ["rg", "-n", query, str(self.root)],
                check=False,
                capture_output=True,
                text=True,
            )
            ok = proc.returncode in (0, 1)
            output = (proc.stdout or proc.stderr).strip()
            return ToolResult(self.name, args, ok, output)
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class ReadFileTool:
    name = "read_file"
    description = "Read a text file from repository."

    def __init__(self, root: Path):
        self.root = root

    def run(self, args: Dict[str, Any]) -> ToolResult:
        rel_path = str(args.get("path", "")).strip()
        if not rel_path:
            return ToolResult(self.name, args, False, "", "missing path")

        target = (self.root / rel_path).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            return ToolResult(self.name, args, False, "", "path escapes repository root")
        if not target.exists():
            return ToolResult(self.name, args, False, "", f"file not found: {target}")

        try:
            text = target.read_text(encoding="utf-8")
            max_chars = int(args.get("max_chars", 4000))
            return ToolResult(self.name, args, True, text[:max_chars])
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class RunTestsTool:
    name = "run_tests"
    description = "Run unit tests with unittest discovery."

    def __init__(self, root: Path):
        self.root = root

    def run(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", "test*.py"))
        try:
            proc = subprocess.run(
                ["python", "-m", "unittest", "discover", "-s", "tests", "-p", pattern],
                cwd=self.root,
                check=False,
                capture_output=True,
                text=True,
            )
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return ToolResult(self.name, args, proc.returncode == 0, output)
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class CommandTool:
    name = "run_command"
    description = "Run a shell command with allowlisted prefixes."

    def __init__(self, root: Path, allowed_prefixes: Optional[List[str]] = None):
        self.root = root
        self.allowed_prefixes = allowed_prefixes or ["rg", "ls", "cat", "python", "pytest"]

    def run(self, args: Dict[str, Any]) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult(self.name, args, False, "", "missing command")

        cmd_prefix = command.split()[0]
        if cmd_prefix not in self.allowed_prefixes:
            return ToolResult(
                self.name,
                args,
                False,
                "",
                f"command prefix not allowed: {cmd_prefix}",
            )

        try:
            proc = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return ToolResult(self.name, args, proc.returncode == 0, output)
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class RAGSearchTool:
    name = "rag_search"
    description = "Search memory for context based on a query."

    def __init__(self, memory_manager):
        self.memory = memory_manager

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(self.name, args, False, "", "missing query")
        try:
            context = self.memory.get_relevant_context(query)
            return ToolResult(self.name, args, True, context)
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class TrainModelTool:
    name = "train_model"
    description = "Trigger model training on a given dataset."

    def __init__(self, training_callback=None):
        self.training_callback = training_callback

    def run(self, args: Dict[str, Any]) -> ToolResult:
        dataset = str(args.get("dataset", "")).strip()
        if not dataset:
            return ToolResult(self.name, args, False, "", "missing dataset param")
        try:
            if self.training_callback:
                self.training_callback(dataset)
            return ToolResult(self.name, args, True, f"Training triggered on {dataset}")
        except Exception as exc:
            return ToolResult(self.name, args, False, "", str(exc))


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())


class AgentCoworker:
    """
    Minimal planner-executor-critic loop for repository coworking tasks.
    """

    def __init__(self, registry: ToolRegistry, max_retries: int = 2, inference_engine=None):
        self.registry = registry
        self.max_retries = max_retries
        self.trace: List[ToolResult] = []
        self.inference_engine = inference_engine
        self.thought_process: str = ""

    def _plan_with_llm(self, user_query: str) -> List[ToolCall]:
        tools_info = []
        for name in self.registry.list_tools():
            tool = self.registry.get(name)
            if tool:
                tools_info.append(f"- {name}: {tool.description}")
        tools_desc = "\n".join(tools_info)
        prompt = f"""You are an autonomous planning agent. Available tools:
{tools_desc}
User query: {user_query}

First, think step-by-step about what tools to use and why.
Then, respond with a JSON array of tool calls inside a ```json block. 
Example:
I need to search the repo to find the file.
```json
[{{"tool_name": "repo_search", "args": {{"query": "User"}}}}]
```
"""
        try:
            # Check if inference engine is capable of MCTS (System 2)
            if hasattr(self.inference_engine, 'reward_model') and hasattr(self.inference_engine, '_lm_generate_multiple'):
                from src.core.cognitive.reasoning_engine import ReasoningEngine
                engine = ReasoningEngine(
                    language_model_generate=self.inference_engine._lm_generate_multiple,
                    reward_evaluator=self.inference_engine.reward_model.evaluate,
                    max_depth=3, # Shallow depth to avoid infinite loops in JSON planning
                    simulations_per_step=3,
                    branching_factor=2
                )
                response = engine.search(prompt)
                self.thought_process = response
            else:
                response_chunks = list(self.inference_engine.generate_stream(prompt, max_new_tokens=1024))
                response = "".join(response_chunks).strip()
                self.thought_process = response

            # Clean MCTS artifacts (Step N:, <thought> tags) from the response
            import re as _re
            clean_response = response
            # Remove Step N: prefixes from MCTS thought chains
            clean_response = _re.sub(r'Step \d+:\s*', '', clean_response)
            # Remove <thought>...</thought> tags
            clean_response = _re.sub(r'</?thought>', '', clean_response)

            json_text = clean_response
            if "```json" in clean_response:
                json_text = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                json_text = clean_response.split("```")[1].split("```")[0].strip()
            # Try to find a JSON array even without code fences
            elif "[" in clean_response:
                start = clean_response.index("[")
                end = clean_response.rindex("]") + 1
                json_text = clean_response[start:end]
            
            calls = json.loads(json_text)
            if isinstance(calls, list):
                tool_calls = []
                for call in calls:
                    tool_calls.append(ToolCall(tool_name=call.get("tool_name", ""), args=call.get("args", {})))
                return tool_calls
        except Exception:
            pass
        return []

    def _plan(self, user_query: str) -> List[ToolCall]:
        if self.inference_engine:
            calls = self._plan_with_llm(user_query)
            if calls:
                return calls

        query = user_query.lower()
        if "test" in query:
            return [ToolCall("run_tests", {"pattern": "test*.py"})]
        if "search" in query or "find" in query:
            return [ToolCall("repo_search", {"query": user_query})]
        if "read" in query and ".py" in query:
            return [ToolCall("read_file", {"path": user_query.split()[-1]})]
        if "train" in query:
            return [ToolCall("train_model", {"dataset": "auto"})]
        if "memory" in query or "rag" in query or "context" in query:
            return [ToolCall("rag_search", {"query": user_query})]
        return [ToolCall("repo_search", {"query": user_query})]

    def _critic_should_retry(self, result: ToolResult, attempt: int) -> bool:
        return (not result.ok) and attempt < self.max_retries

    def run(self, user_query: str) -> Dict[str, Any]:
        self.trace = []
        self.thought_process = ""
        planned_calls = self._plan(user_query)

        for call in planned_calls:
            tool = self.registry.get(call.tool_name)
            if tool is None:
                result = ToolResult(
                    tool_name=call.tool_name,
                    args=call.args,
                    ok=False,
                    result="",
                    error="tool not registered",
                )
                self.trace.append(result)
                continue

            attempt = 0
            while True:
                result = tool.run(call.args)
                self.trace.append(result)
                if not self._critic_should_retry(result, attempt):
                    break
                attempt += 1

        return {
            "query": user_query,
            "planned_calls": [asdict(c) for c in planned_calls],
            "trace": [asdict(r) for r in self.trace],
            "thought_process": self.thought_process,
            "success": all(item.ok for item in self.trace) if self.trace else False,
        }
