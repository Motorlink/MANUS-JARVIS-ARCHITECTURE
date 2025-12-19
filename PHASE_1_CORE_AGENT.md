# Phase 1: Core Agent - Detaillierte Implementierung

**Dauer:** Wochen 2-5 (4 Wochen)  
**Ziel:** Funktionsfähiger KI-Agent mit LLM, Agent Loop, Task Planning und Function Calling

---

## 📋 Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [1.1 LLM Integration](#11-llm-integration)
3. [1.2 Agent Loop](#12-agent-loop)
4. [1.3 Task Planning](#13-task-planning)
5. [1.4 Function Calling](#14-function-calling)
6. [1.5 Basic File Operations](#15-basic-file-operations)
7. [1.6 Shell Execution](#16-shell-execution)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Übersicht

Am Ende von Phase 1 hast du:
- ✅ Funktionierenden OpenAI GPT-4o Client
- ✅ Agent Loop mit Reasoning
- ✅ Task Planning System
- ✅ Function Calling Framework
- ✅ Basis-Tools (File, Shell)
- ✅ Erste End-to-End Tests

---

## 1.1 LLM Integration

### Setup

**1. Projektstruktur erstellen:**
```bash
mkdir manus-jarvis
cd manus-jarvis
mkdir -p backend/{core,tools,utils}
touch backend/__init__.py
touch backend/core/__init__.py
touch backend/tools/__init__.py
touch backend/utils/__init__.py
```

**2. Dependencies installieren:**
```bash
# requirements.txt
openai>=1.0.0
tiktoken>=0.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

```bash
pip install -r requirements.txt
```

**3. Environment Variables:**
```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
MAX_TOKENS=4096
TEMPERATURE=0.7
```

---

### LLM Client Implementation

**`backend/core/llm_client.py`:**
```python
"""
LLM Client - OpenAI GPT-4o Integration
"""
import os
from typing import List, Dict, Optional, Any
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """OpenAI GPT-4o Client with streaming and token counting"""
    
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.encoding_for_model(self.model)
        
        # Statistics
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on token usage"""
        # GPT-4o pricing (as of Dec 2024)
        prompt_cost = prompt_tokens * 0.00001  # $0.01 per 1K tokens
        completion_cost = completion_tokens * 0.00003  # $0.03 per 1K tokens
        return prompt_cost + completion_cost
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat request to OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            tool_choice: "auto", "none", or specific tool
            stream: Whether to stream the response
        
        Returns:
            Response dict with 'content', 'tool_calls', 'usage'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
            if stream:
                return self._handle_stream(response)
            else:
                return self._handle_response(response)
        
        except Exception as e:
            return {
                "error": str(e),
                "content": None,
                "tool_calls": None
            }
    
    def _handle_response(self, response) -> Dict[str, Any]:
        """Handle non-streaming response"""
        choice = response.choices[0]
        message = choice.message
        
        # Update statistics
        usage = response.usage
        self.total_tokens_used += usage.total_tokens
        cost = self.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        self.total_cost += cost
        
        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost": cost
            }
        }
    
    def _handle_stream(self, response) -> Dict[str, Any]:
        """Handle streaming response"""
        content = ""
        tool_calls = []
        
        for chunk in response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)
            
            if delta.tool_calls:
                tool_calls.extend(delta.tool_calls)
        
        print()  # Newline after stream
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": "stop"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": round(self.total_cost, 4),
            "model": self.model
        }


# Example usage
if __name__ == "__main__":
    client = LLMClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = client.chat(messages)
    print(f"Response: {response['content']}")
    print(f"Usage: {response['usage']}")
    print(f"Stats: {client.get_stats()}")
```

---

### Testing LLM Client

**`tests/test_llm_client.py`:**
```python
import pytest
from backend.core.llm_client import LLMClient


def test_llm_client_basic():
    """Test basic LLM client functionality"""
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Say 'Hello World'"}
    ]
    
    response = client.chat(messages)
    
    assert response["content"] is not None
    assert "hello" in response["content"].lower()
    assert response["usage"]["total_tokens"] > 0


def test_token_counting():
    """Test token counting"""
    client = LLMClient()
    
    text = "Hello, how are you?"
    tokens = client.count_tokens(text)
    
    assert tokens > 0
    assert tokens < 10  # Should be around 5-6 tokens


def test_cost_estimation():
    """Test cost estimation"""
    client = LLMClient()
    
    cost = client.estimate_cost(prompt_tokens=1000, completion_tokens=500)
    
    assert cost > 0
    assert cost < 1  # Should be a few cents
```

**Run tests:**
```bash
pytest tests/test_llm_client.py -v
```

---

## 1.2 Agent Loop

### Agent Loop Implementation

**`backend/core/agent_loop.py`:**
```python
"""
Agent Loop - Core reasoning engine
"""
from typing import List, Dict, Any, Optional
from backend.core.llm_client import LLMClient
from backend.core.tool_registry import ToolRegistry


class AgentLoop:
    """
    Core agent loop implementing:
    1. Analyze Context
    2. Think (LLM)
    3. Select Tool
    4. Execute
    5. Observe
    6. Iterate
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_iterations: int = 10
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        
        # Context
        self.messages = []
        self.iteration_count = 0
        self.task_complete = False
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Run agent loop for given task
        
        Args:
            task: User task description
        
        Returns:
            Result dict with 'success', 'result', 'iterations'
        """
        # Initialize
        self.messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": task
            }
        ]
        self.iteration_count = 0
        self.task_complete = False
        
        # Agent loop
        while not self.task_complete and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration_count}")
            print(f"{'='*60}\n")
            
            # 1. Think (LLM call with tools)
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools.get_schemas(),
                tool_choice="auto"
            )
            
            # 2. Check if task is complete
            if response["finish_reason"] == "stop":
                # No tool calls - task complete
                self.task_complete = True
                return {
                    "success": True,
                    "result": response["content"],
                    "iterations": self.iteration_count,
                    "stats": self.llm.get_stats()
                }
            
            # 3. Execute tool calls
            if response["tool_calls"]:
                self._execute_tools(response["tool_calls"])
            else:
                # No tool calls but not finished - error
                return {
                    "success": False,
                    "error": "No tool calls but task not complete",
                    "iterations": self.iteration_count
                }
        
        # Max iterations reached
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "iterations": self.iteration_count
        }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for agent"""
        return """You are an autonomous AI agent.

Your goal is to complete the user's task using the available tools.

Process:
1. Analyze the task
2. Decide which tool(s) to use
3. Execute tools
4. Observe results
5. Repeat until task is complete

When the task is complete, provide a final answer without calling any tools.

Available tools:
- Use tools to gather information, perform actions, etc.
- Always check tool results before proceeding
- If a tool fails, try an alternative approach

Be efficient and complete the task in as few iterations as possible."""
    
    def _execute_tools(self, tool_calls: List[Any]) -> None:
        """Execute tool calls and add results to context"""
        # Add assistant message with tool calls
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # Execute each tool
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            print(f"🔧 Executing tool: {tool_name}")
            print(f"   Arguments: {tool_args}")
            
            # Execute
            result = self.tools.execute(tool_name, tool_args)
            
            print(f"   Result: {result[:100]}..." if len(str(result)) > 100 else f"   Result: {result}")
            
            # Add tool result to context
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })


# Example usage
if __name__ == "__main__":
    from backend.core.tool_registry import ToolRegistry
    from backend.tools.basic_tools import register_basic_tools
    
    # Setup
    llm = LLMClient()
    registry = ToolRegistry()
    register_basic_tools(registry)
    
    # Run agent
    agent = AgentLoop(llm, registry)
    result = agent.run("What is 15 * 23?")
    
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Result: {result.get('result', result.get('error'))}")
    print(f"Iterations: {result['iterations']}")
    print(f"Stats: {result.get('stats')}")
```

---

## 1.3 Task Planning

### Task Planner Implementation

**`backend/core/task_planner.py`:**
```python
"""
Task Planner - Hierarchical task planning
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class TaskPlanner:
    """
    Hierarchical task planner
    
    Creates and manages multi-phase task plans
    """
    
    def __init__(self):
        self.current_plan = None
        self.current_phase_id = None
    
    def create_plan(
        self,
        goal: str,
        phases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create new task plan
        
        Args:
            goal: Overall goal description
            phases: List of phase dicts with 'id', 'title', 'capabilities'
        
        Returns:
            Plan dict
        """
        self.current_plan = {
            "goal": goal,
            "phases": phases,
            "current_phase_id": phases[0]["id"] if phases else None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.current_phase_id = self.current_plan["current_phase_id"]
        
        return self.current_plan
    
    def advance_phase(self) -> Optional[Dict[str, Any]]:
        """
        Advance to next phase
        
        Returns:
            Next phase dict or None if no more phases
        """
        if not self.current_plan:
            raise ValueError("No active plan")
        
        phases = self.current_plan["phases"]
        current_idx = self._get_phase_index(self.current_phase_id)
        
        if current_idx is None:
            raise ValueError(f"Current phase {self.current_phase_id} not found")
        
        # Check if there's a next phase
        if current_idx + 1 < len(phases):
            next_phase = phases[current_idx + 1]
            self.current_phase_id = next_phase["id"]
            self.current_plan["current_phase_id"] = next_phase["id"]
            self.current_plan["updated_at"] = datetime.now().isoformat()
            return next_phase
        else:
            # No more phases
            return None
    
    def update_plan(
        self,
        goal: Optional[str] = None,
        phases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Update existing plan
        
        Args:
            goal: New goal (optional)
            phases: New phases (optional)
        
        Returns:
            Updated plan dict
        """
        if not self.current_plan:
            raise ValueError("No active plan to update")
        
        if goal:
            self.current_plan["goal"] = goal
        
        if phases:
            self.current_plan["phases"] = phases
            # Reset to first phase
            self.current_phase_id = phases[0]["id"]
            self.current_plan["current_phase_id"] = self.current_phase_id
        
        self.current_plan["updated_at"] = datetime.now().isoformat()
        
        return self.current_plan
    
    def get_current_phase(self) -> Optional[Dict[str, Any]]:
        """Get current phase"""
        if not self.current_plan or not self.current_phase_id:
            return None
        
        phases = self.current_plan["phases"]
        for phase in phases:
            if phase["id"] == self.current_phase_id:
                return phase
        
        return None
    
    def _get_phase_index(self, phase_id: int) -> Optional[int]:
        """Get index of phase by ID"""
        phases = self.current_plan["phases"]
        for idx, phase in enumerate(phases):
            if phase["id"] == phase_id:
                return idx
        return None
    
    def save_plan(self, filepath: str) -> None:
        """Save plan to JSON file"""
        if not self.current_plan:
            raise ValueError("No active plan to save")
        
        with open(filepath, 'w') as f:
            json.dump(self.current_plan, f, indent=2)
    
    def load_plan(self, filepath: str) -> Dict[str, Any]:
        """Load plan from JSON file"""
        with open(filepath, 'r') as f:
            self.current_plan = json.load(f)
        
        self.current_phase_id = self.current_plan["current_phase_id"]
        
        return self.current_plan


# Example usage
if __name__ == "__main__":
    planner = TaskPlanner()
    
    # Create plan
    plan = planner.create_plan(
        goal="Research and write report on AI agents",
        phases=[
            {
                "id": 1,
                "title": "Research AI agents",
                "capabilities": ["deep_research"]
            },
            {
                "id": 2,
                "title": "Analyze findings",
                "capabilities": ["data_analysis"]
            },
            {
                "id": 3,
                "title": "Write report",
                "capabilities": ["technical_writing"]
            }
        ]
    )
    
    print("Plan created:")
    print(json.dumps(plan, indent=2))
    
    # Get current phase
    current = planner.get_current_phase()
    print(f"\nCurrent phase: {current['title']}")
    
    # Advance
    next_phase = planner.advance_phase()
    print(f"Advanced to: {next_phase['title']}")
    
    # Save
    planner.save_plan("task_plan.json")
    print("\nPlan saved to task_plan.json")
```

---

## 1.4 Function Calling

### Tool Registry Implementation

**`backend/core/tool_registry.py`:**
```python
"""
Tool Registry - Manages all available tools
"""
from typing import Dict, Any, Callable, List
import json


class ToolRegistry:
    """
    Central registry for all tools
    
    Handles tool registration, schema management, and execution
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        function: Callable,
        schema: Dict[str, Any]
    ) -> None:
        """
        Register a tool
        
        Args:
            name: Tool name
            function: Callable function
            schema: OpenAI function schema
        """
        self.tools[name] = {
            "function": function,
            "schema": schema
        }
        
        print(f"✅ Registered tool: {name}")
    
    def execute(self, name: str, arguments: str) -> Any:
        """
        Execute a tool
        
        Args:
            name: Tool name
            arguments: JSON string of arguments
        
        Returns:
            Tool result
        """
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            # Parse arguments
            args = json.loads(arguments)
            
            # Execute function
            function = self.tools[name]["function"]
            result = function(**args)
            
            return result
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for OpenAI"""
        return [
            {
                "type": "function",
                "function": tool["schema"]
            }
            for tool in self.tools.values()
        ]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())


# Example usage
if __name__ == "__main__":
    registry = ToolRegistry()
    
    # Register a simple tool
    def add(a: int, b: int) -> int:
        return a + b
    
    registry.register(
        name="add",
        function=add,
        schema={
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    )
    
    # Execute
    result = registry.execute("add", '{"a": 5, "b": 3}')
    print(f"Result: {result}")  # 8
    
    # Get schemas
    schemas = registry.get_schemas()
    print(f"Schemas: {json.dumps(schemas, indent=2)}")
```

---

### Basic Tools

**`backend/tools/basic_tools.py`:**
```python
"""
Basic Tools - Calculator, Time, etc.
"""
from datetime import datetime
import math


def get_current_time() -> str:
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> float:
    """
    Calculate mathematical expression
    
    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)")
    
    Returns:
        Result
    """
    try:
        # Safe eval with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    
    except Exception as e:
        return f"Error: {str(e)}"


def register_basic_tools(registry):
    """Register basic tools in registry"""
    
    # get_current_time
    registry.register(
        name="get_current_time",
        function=get_current_time,
        schema={
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    )
    
    # calculate
    registry.register(
        name="calculate",
        function=calculate,
        schema={
            "name": "calculate",
            "description": "Calculate a mathematical expression. Supports basic operations and math functions like sqrt, sin, cos, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14)')"
                    }
                },
                "required": ["expression"]
            }
        }
    )
```

---

## 1.5 Basic File Operations

**`backend/tools/file_tools.py`:**
```python
"""
File Tools - Read, Write, List files
"""
from pathlib import Path
from typing import List


def read_file(filepath: str) -> str:
    """
    Read file content
    
    Args:
        filepath: Path to file
    
    Returns:
        File content
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File '{filepath}' not found"
        
        return path.read_text()
    
    except Exception as e:
        return f"Error: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    """
    Write content to file
    
    Args:
        filepath: Path to file
        content: Content to write
    
    Returns:
        Success message
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Successfully wrote to '{filepath}'"
    
    except Exception as e:
        return f"Error: {str(e)}"


def list_files(directory: str = ".") -> List[str]:
    """
    List files in directory
    
    Args:
        directory: Directory path
    
    Returns:
        List of file paths
    """
    try:
        path = Path(directory)
        if not path.exists():
            return [f"Error: Directory '{directory}' not found"]
        
        files = [str(f) for f in path.iterdir() if f.is_file()]
        return files
    
    except Exception as e:
        return [f"Error: {str(e)}"]


def register_file_tools(registry):
    """Register file tools in registry"""
    
    # read_file
    registry.register(
        name="read_file",
        function=read_file,
        schema={
            "name": "read_file",
            "description": "Read the content of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filepath"]
            }
        }
    )
    
    # write_file
    registry.register(
        name="write_file",
        function=write_file,
        schema={
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    )
    
    # list_files
    registry.register(
        name="list_files",
        function=list_files,
        schema={
            "name": "list_files",
            "description": "List all files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path (default: current directory)"
                    }
                },
                "required": []
            }
        }
    )
```

---

## 1.6 Shell Execution

**`backend/tools/shell_tools.py`:**
```python
"""
Shell Tools - Execute shell commands
"""
import subprocess
from typing import Dict, Any


def execute_shell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute shell command
    
    Args:
        command: Shell command to execute
        timeout: Timeout in seconds
    
    Returns:
        Dict with 'stdout', 'stderr', 'returncode'
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout} seconds",
            "success": False
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def register_shell_tools(registry):
    """Register shell tools in registry"""
    
    registry.register(
        name="execute_shell",
        function=execute_shell,
        schema={
            "name": "execute_shell",
            "description": "Execute a shell command. Use with caution!",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)"
                    }
                },
                "required": ["command"]
            }
        }
    )
```

---

## Testing

### End-to-End Test

**`tests/test_agent_e2e.py`:**
```python
"""
End-to-end test for Phase 1
"""
from backend.core.llm_client import LLMClient
from backend.core.agent_loop import AgentLoop
from backend.core.tool_registry import ToolRegistry
from backend.tools.basic_tools import register_basic_tools
from backend.tools.file_tools import register_file_tools
from backend.tools.shell_tools import register_shell_tools


def test_agent_calculation():
    """Test agent with calculation"""
    llm = LLMClient()
    registry = ToolRegistry()
    register_basic_tools(registry)
    
    agent = AgentLoop(llm, registry)
    result = agent.run("What is 15 * 23?")
    
    assert result["success"]
    assert "345" in result["result"]


def test_agent_file_operations():
    """Test agent with file operations"""
    llm = LLMClient()
    registry = ToolRegistry()
    register_basic_tools(registry)
    register_file_tools(registry)
    
    agent = AgentLoop(llm, registry)
    result = agent.run("Write 'Hello World' to test.txt and then read it back")
    
    assert result["success"]
    assert "Hello World" in result["result"]


def test_agent_shell():
    """Test agent with shell execution"""
    llm = LLMClient()
    registry = ToolRegistry()
    register_basic_tools(registry)
    register_shell_tools(registry)
    
    agent = AgentLoop(llm, registry)
    result = agent.run("List all files in the current directory using ls command")
    
    assert result["success"]


if __name__ == "__main__":
    print("Running E2E tests...\n")
    
    print("Test 1: Calculation")
    test_agent_calculation()
    print("✅ Passed\n")
    
    print("Test 2: File Operations")
    test_agent_file_operations()
    print("✅ Passed\n")
    
    print("Test 3: Shell Execution")
    test_agent_shell()
    print("✅ Passed\n")
    
    print("All tests passed! 🎉")
```

---

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: OPENAI_API_KEY not found in environment
```
**Solution:** Create `.env` file with your API key

**2. Token Limit Exceeded**
```
Error: This model's maximum context length is 128000 tokens
```
**Solution:** Reduce `max_tokens` or implement context pruning

**3. Tool Not Found**
```
Error: Tool 'xyz' not found
```
**Solution:** Make sure tool is registered before agent.run()

**4. Timeout Error**
```
Error: Command timed out after 30 seconds
```
**Solution:** Increase timeout parameter

---

## Next Steps

After completing Phase 1, you should have:
- ✅ Working LLM client
- ✅ Functional agent loop
- ✅ Task planning system
- ✅ Function calling framework
- ✅ Basic tools (calc, file, shell)

**Move to Phase 2:** Multimodal Processing (Images, Audio, PDFs)

---

**Phase 1 Complete!** 🎉

---

**Version:** 1.0  
**Last Updated:** December 19, 2024
