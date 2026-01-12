"""REPL Environment for RLM - Allows boss to execute Python code."""
import sys
from io import StringIO
from typing import Dict, Any, Callable


class REPLEnvironment:
    """
    Python REPL environment where the boss LM can execute code.
    Context is stored as a variable, not passed in prompts.
    """
    
    def __init__(self):
        """Initialize the REPL environment."""
        self.namespace = {}
        self.output_buffer = []
        
    def add_function(self, name: str, func: Callable):
        """Add a function to the REPL namespace."""
        self.namespace[name] = func
    
    def add_variable(self, name: str, value: Any):
        """Add a variable to the REPL namespace."""
        self.namespace[name] = value
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code in the REPL environment."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        result = {
            "success": True,
            "output": "",
            "error": None,
            "variables_added": []
        }
        
        try:
            vars_before = set(self.namespace.keys())
            exec(code, self.namespace)
            vars_after = set(self.namespace.keys())
            result["variables_added"] = list(vars_after - vars_before)
            
            output = sys.stdout.getvalue()
            if output:
                result["output"] = output
                self.output_buffer.append(output)
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["output"] = sys.stdout.getvalue()
            
        finally:
            sys.stdout = old_stdout
        
        return result
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self.namespace.get(name)
    
    def get_output_history(self) -> list:
        """Get all output from executed code."""
        return self.output_buffer.copy()
    
    def clear(self):
        """Clear the REPL environment."""
        self.namespace = {}
        self.output_buffer = []


class SafeREPLEnvironment(REPLEnvironment):
    """
    Safer REPL environment with restricted operations.
    Blocks dangerous operations like file I/O, network, etc.
    """
    
    BLOCKED_IMPORTS = {
        'os', 'sys', 'subprocess', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib',
        'pickle', 'shelve', 'marshal'
    }
    
    BLOCKED_BUILTINS = {
        'open', 'eval', 'exec', 'compile',
        '__import__', 'globals', 'locals', 'vars'
    }
    
    def __init__(self):
        """Initialize safe REPL with restricted builtins."""
        super().__init__()
        
        # Create safe builtins
        safe_builtins = {
            'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'map': map, 'filter': filter, 'sum': sum,
            'min': min, 'max': max, 'sorted': sorted, 'print': print,
            'any': any, 'all': all,  # CRITICAL: Needed for list comprehensions!
            'isinstance': isinstance, 'type': type,  # Useful for type checking
            'abs': abs, 'round': round,  # Math functions
        }
        
        self.namespace['__builtins__'] = safe_builtins
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code with safety checks."""
        # Check for blocked imports
        for blocked in self.BLOCKED_IMPORTS:
            if f'import {blocked}' in code or f'from {blocked}' in code:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Import of '{blocked}' is not allowed",
                    "variables_added": []
                }
        
        # Check for blocked operations
        for blocked in self.BLOCKED_BUILTINS:
            if blocked in code:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Use of '{blocked}' is not allowed",
                    "variables_added": []
                }
        
        return super().execute(code)