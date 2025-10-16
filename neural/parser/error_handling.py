"""
Enhanced error handling for the Neural parser.
Provides detailed, context-aware error messages and recovery strategies.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from lark import UnexpectedToken, UnexpectedCharacters
import difflib

@dataclass
class ParserError:
    """Structured representation of a parsing error."""
    message: str
    line: int
    column: int
    context: str
    suggestion: Optional[str] = None

class NeuralParserError(Exception):
    """Custom exception class for Neural parser errors."""
    def __init__(self, error: ParserError):
        self.error = error
        super().__init__(str(error))

class ErrorHandler:
    """Handles and formats parser errors with helpful context and suggestions."""
    
    COMMON_MISTAKES = {
        "Dense": ["dense", "Dence", "DNse"],
        "Conv2D": ["conv2d", "Conv2d", "conv2D"],
        "Input": ["input", "input_layer", "Iput"],
        "Output": ["output", "output_layer", "Ouput"],
        "activation": ["activate", "activations", "activ"],
    }
    
    @staticmethod
    def get_line_context(code: str, line_no: int, context_lines: int = 2) -> str:
        """Get the surrounding lines of code for context."""
        lines = code.splitlines()
        start = max(0, line_no - context_lines)
        end = min(len(lines), line_no + context_lines + 1)
        
        context = []
        for i in range(start, end):
            prefix = "-> " if i == line_no else "   "
            context.append(f"{prefix}{i+1:4d} | {lines[i]}")
        return "\n".join(context)
    
    @staticmethod
    def suggest_correction(token: str) -> Optional[str]:
        """Suggest corrections for common mistakes."""
        for correct, mistakes in ErrorHandler.COMMON_MISTAKES.items():
            if token.lower() in [m.lower() for m in mistakes]:
                return correct
                
        # Use difflib for more general suggestions
        all_valid_tokens = list(ErrorHandler.COMMON_MISTAKES.keys())
        matches = difflib.get_close_matches(token, all_valid_tokens, n=1)
        return matches[0] if matches else None
    
    @classmethod
    def handle_unexpected_token(cls, error: UnexpectedToken, code: str) -> ParserError:
        """Handle unexpected token errors with context."""
        context = cls.get_line_context(code, error.line - 1)
        suggestion = cls.suggest_correction(str(error.token))
        
        msg = f"Unexpected token '{error.token}' at line {error.line}, column {error.column}."
        if suggestion:
            msg += f" Did you mean '{suggestion}'?"
            
        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context,
            suggestion=suggestion
        )
    
    @classmethod
    def handle_unexpected_char(cls, error: UnexpectedCharacters, code: str) -> ParserError:
        """Handle unexpected character errors with context."""
        context = cls.get_line_context(code, error.line - 1)
        
        msg = f"Unexpected character '{error.char}' at line {error.line}, column {error.column}."
        allowed = error.allowed if hasattr(error, 'allowed') else None
        if allowed:
            msg += f" Expected one of: {', '.join(allowed)}"
            
        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context
        )
    
    @classmethod
    def handle_shape_error(cls, shape_error: Exception, code: str, line_no: int) -> ParserError:
        """Handle shape propagation errors with context."""
        context = cls.get_line_context(code, line_no)
        
        return ParserError(
            message=str(shape_error),
            line=line_no,
            column=0,  # Shape errors typically affect whole line
            context=context
        )