import pytest
from harmony.tools.tool_registry import ToolRegistry, registry

def test_tool_registry():
    tr = ToolRegistry()
    
    # Register tool
    def add_nums(a, b):
        return a + b
        
    tr.register_tool("add", add_nums)
    assert "add" in tr.tools
    
    # Execute tool
    res = tr.execute_tool("add", a=5, b=3)
    assert res == 8
    
    # Execute non-existent tool should raise ValueError
    with pytest.raises(ValueError, match="Tool sub not found."):
        tr.execute_tool("sub", a=5, b=3)

def test_default_registry_tools():
    # Test built-in tools in global registry
    assert "python" in registry.tools
    assert "search" in registry.tools
    
    py_res = registry.execute_tool("python", code="print('hello')")
    assert py_res == "mock execution result"
    
    search_res = registry.execute_tool("search", query="harmony architecture")
    assert search_res == "mock search results"
