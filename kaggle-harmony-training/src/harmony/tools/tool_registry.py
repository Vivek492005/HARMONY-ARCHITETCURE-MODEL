from typing import Callable, Dict, Any, List

class ToolRegistry:
    """
    Phase E: Tool Use System.
    Native tool calling registry.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        
    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func
        
    def execute_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found.")
        return self.tools[name](**kwargs)
        
# Example default tools
def mock_python_execution(code: str) -> str:
    return "mock execution result"

def mock_web_search(query: str) -> str:
    return "mock search results"

registry = ToolRegistry()
registry.register_tool("python", mock_python_execution)
registry.register_tool("search", mock_web_search)
