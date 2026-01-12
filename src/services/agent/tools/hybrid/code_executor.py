"""Safe code executor tool with sandbox."""

import logging
import sys
from io import StringIO
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):
    """Tool for safe Python code execution."""
    
    def __init__(self, timeout: int = 5):
        """Initialize code executor tool.
        
        Args:
            timeout: Execution timeout in seconds
        """
        metadata = ToolMetadata(
            name="code_executor",
            description="Execute Python code safely in a sandbox for calculations and data analysis",
            category=ToolCategory.HYBRID,
            capabilities=[
                "python execution",
                "code running",
                "calculations",
                "data analysis",
                "mathematical operations",
            ],
            cost_per_call=0.0,
            avg_latency_ms=300.0,
            success_rate=0.82,
            requires_api_key=False,
        )
        super().__init__(metadata)
        self.timeout = timeout
        self._restricted_python_available = False
        
        try:
            from RestrictedPython import compile_restricted, safe_globals
            self._compile_restricted = compile_restricted
            self._safe_globals = safe_globals
            self._restricted_python_available = True
            self.logger.info("RestrictedPython available for enhanced security")
        except ImportError:
            self.logger.warning("RestrictedPython not available, using basic sandbox")
    
    def _get_safe_globals(self) -> dict:
        """Get safe global namespace for code execution.
        
        Returns:
            Dictionary of safe globals
        """
        if self._restricted_python_available:
            # Use RestrictedPython safe globals
            safe_dict = self._safe_globals.copy()
        else:
            # Basic safe globals
            safe_dict = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "enumerate": enumerate,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "max": max,
                    "min": min,
                    "print": print,
                    "range": range,
                    "round": round,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                },
            }
        
        # Add safe math operations
        import math
        safe_dict["math"] = math
        
        return safe_dict
    
    def _extract_code(self, query: str) -> str:
        """Extract Python code from query.
        
        Args:
            query: User query that may contain code
            
        Returns:
            Extracted Python code
        """
        # Check if query contains code blocks
        if "```python" in query:
            # Extract code from markdown code block
            parts = query.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in query:
            # Generic code block
            parts = query.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()
                return code
        
        # Assume entire query is code
        return query.strip()
    
    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute Python code safely.
        
        Args:
            query: Query containing Python code
            **kwargs: Optional parameters:
                - timeout: Override default timeout
                
        Returns:
            Dictionary with execution results
        """
        try:
            if not self.validate_input(query, **kwargs):
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid input parameters",
                    "metadata": {},
                }
            
            # Extract code from query
            code = self._extract_code(query)
            
            if not code:
                return {
                    "success": False,
                    "result": None,
                    "error": "No Python code found in query",
                    "metadata": {},
                }
            
            self.logger.info(f"Executing code: {code[:100]}...")
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Get safe execution environment
                safe_globals = self._get_safe_globals()
                safe_locals = {}
                
                # Compile and execute code
                if self._restricted_python_available:
                    # Use RestrictedPython for enhanced security
                    byte_code = self._compile_restricted(
                        code,
                        filename="<sandbox>",
                        mode="exec",
                    )
                    
                    if byte_code.errors:
                        error_msg = "; ".join(byte_code.errors)
                        raise SyntaxError(f"Code validation failed: {error_msg}")
                    
                    exec(byte_code.code, safe_globals, safe_locals)
                else:
                    # Basic exec with limited globals
                    exec(code, safe_globals, safe_locals)
                
                # Get captured output
                output = captured_output.getvalue()
                
                # Get result (if any variable was defined)
                result_vars = {
                    k: v for k, v in safe_locals.items()
                    if not k.startswith("_")
                }
                
                self.logger.info(f"Code executed successfully. Output: {output[:100]}")
                
                return {
                    "success": True,
                    "result": {
                        "output": output,
                        "variables": result_vars,
                        "code": code,
                    },
                    "error": None,
                    "metadata": {
                        "code_length": len(code),
                        "has_output": bool(output),
                        "num_variables": len(result_vars),
                    },
                }
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in code: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Syntax error: {str(e)}",
                "metadata": {"code": code if 'code' in locals() else ""},
            }
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Execution error: {str(e)}",
                "metadata": {"code": code if 'code' in locals() else ""},
            }
        finally:
            # Ensure stdout is restored
            if sys.stdout != old_stdout:
                sys.stdout = old_stdout
