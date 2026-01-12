"""Tool registry for managing available tools."""

import logging
from typing import Optional

from src.services.agent.tools.base import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already exists
        """
        name = tool.metadata.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        self._tools[name] = tool
        self.logger.info(
            f"Registered tool: {name} (category={tool.metadata.category}, "
            f"version={tool.metadata.version})"
        )
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> list[BaseTool]:
        """List all tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tools
        """
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.metadata.category == category]
        return tools
    
    def get_tool_by_capability(self, capability: str) -> list[BaseTool]:
        """Find tools by capability keyword.
        
        Args:
            capability: Capability keyword to search
            
        Returns:
            List of tools with matching capability
        """
        matching_tools = []
        capability_lower = capability.lower()
        
        for tool in self._tools.values():
            for cap in tool.metadata.capabilities:
                if capability_lower in cap.lower():
                    matching_tools.append(tool)
                    break
        
        return matching_tools
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self.logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_all_tool_names(self) -> list[str]:
        """Get all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={len(self._tools)})"


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.
    
    Returns:
        Global tool registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
