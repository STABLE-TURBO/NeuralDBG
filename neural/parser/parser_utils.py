"""Parser utility functions for Neural DSL.

This module contains utility functions for error handling, parsing,
and grammar creation that are used by the main parser.
"""

import logging
from typing import Any, Dict, List, Optional

import lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

from .error_handling import ErrorHandler
from .hpo_utils import Severity


logger = logging.getLogger('neural.parser')


def log_by_severity(severity: Severity, message: str) -> None:
    """Log a message based on its severity level.
    
    Args:
        severity: Severity level
        message: Message to log
    """
    if severity == Severity.DEBUG:
        logger.debug(message)
    elif severity == Severity.INFO:
        logger.info(message)
    elif severity == Severity.WARNING:
        logger.warning(message)
    elif severity == Severity.ERROR:
        logger.error(message)
    elif severity == Severity.CRITICAL:
        logger.critical(message)


class DSLValidationError(Exception):
    """Exception raised for validation errors in DSL parsing.
    
    Attributes:
        severity: The severity level of the validation error
        line: The line number where the error occurred
        column: The column number where the error occurred
        message: The raw error message
    """

    def __init__(self, message: str, severity: Severity = Severity.ERROR,
                 line: Optional[int] = None, column: Optional[int] = None):
        self.severity = severity
        self.line = line
        self.column = column
        self.message = message

        if line and column:
            super().__init__(f"{severity.name} at line {line}, column {column}: {message}")
        else:
            super().__init__(f"{severity.name}: {message}")


def custom_error_handler(error: Exception) -> Dict[str, Any]:
    """Handle Lark parsing errors and convert them to DSLValidationError.
    
    Args:
        error: The error to handle
        
    Returns:
        Dictionary with warning information (if recoverable)
        
    Raises:
        DSLValidationError: For non-recoverable errors
    """
    if isinstance(error, KeyError):
        msg = "Unexpected end of input (KeyError). The parser did not expect '$END'."
        severity = Severity.ERROR
        line = column = None
    elif isinstance(error, UnexpectedCharacters):
        msg = (f"Syntax error at line {error.line}, column {error.column}: "
               f"Unexpected character '{error.char}'.\n"
               f"Expected one of: {', '.join(sorted(error.allowed))}")
        severity = Severity.ERROR
        line, column = error.line, error.column
    elif isinstance(error, UnexpectedToken):
        # Check for end-of-input scenarios
        if str(error.token) in ['', '$END'] or 'RBRACE' in error.expected:
            msg = "Unexpected end of input - Check for missing closing braces"
            severity = Severity.ERROR
            log_by_severity(severity, msg)
            raise DSLValidationError(msg, severity, error.line, error.column)
        else:
            msg = (f"Syntax error at line {error.line}, column {error.column}: "
                   f"Unexpected token '{error.token}'.\n"
                   f"Expected one of: {', '.join(sorted(error.expected))}")
            severity = Severity.ERROR
        line, column = error.line, error.column
    else:
        msg = str(error)
        severity = Severity.ERROR
        line = getattr(error, 'line', None)
        column = getattr(error, 'column', None)

    log_by_severity(severity, msg)
    if severity.value >= Severity.ERROR.value:
        raise DSLValidationError(msg, severity, line, column)
    return {"warning": msg, "line": line, "column": column}


def safe_parse(parser: lark.Lark, text: str) -> Dict[str, Any]:
    """Safely parse text using the provided parser.
    
    Args:
        parser: The Lark parser to use
        text: The input text to parse
        
    Returns:
        Dictionary containing 'result' (parse tree) and 'warnings' (list)
        
    Raises:
        DSLValidationError: If parsing fails
    """
    warnings = []

    # Tokenize and log
    logger.debug("Token stream:")
    tokens = list(parser.lex(text))
    for token in tokens:
        logger.debug(f"Token: {token.type}('{token.value}') at line {token.line}, column {token.column}")

    try:
        tree = parser.parse(text)
        logger.debug("Parse successful, tree generated.")
        return {"result": tree, "warnings": warnings}
    except (UnexpectedCharacters, UnexpectedToken) as e:
        # Use enhanced error handler
        if isinstance(e, UnexpectedToken):
            parser_error = ErrorHandler.handle_unexpected_token(e, text)
        else:
            parser_error = ErrorHandler.handle_unexpected_char(e, text)

        formatted_error = ErrorHandler.format_error(parser_error)
        log_by_severity(Severity.ERROR, formatted_error)
        raise DSLValidationError(parser_error.message, Severity.ERROR,
                                parser_error.line, parser_error.column)
    except DSLValidationError:
        raise
    except Exception as e:
        log_by_severity(Severity.ERROR, f"Unexpected error while parsing: {str(e)}")
        raise DSLValidationError(f"Parser error: {str(e)}", Severity.ERROR)


def split_params(s: str) -> List[str]:
    """Split parameter string by commas, respecting parentheses depth.
    
    Args:
        s: Parameter string to split
        
    Returns:
        List of parameter strings
    """
    parts = []
    current = []
    depth = 0

    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(c)

    if current:
        parts.append(''.join(current).strip())

    return parts
