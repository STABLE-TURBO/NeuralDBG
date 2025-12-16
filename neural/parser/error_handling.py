"""
Enhanced error handling for the Neural parser.
Provides detailed, context-aware error messages and recovery strategies.

This module implements a comprehensive error handling system that:
- Provides precise error location (line/column)
- Shows surrounding code context with visual indicators
- Suggests fixes for common mistakes and typos
- Categorizes errors by severity
- Offers actionable solutions with examples
- Detects common DSL syntax issues
- Provides quick reference guides in error messages

Features:
1. Typo Detection: Identifies common misspellings of layer names and parameters
2. Context-Aware Suggestions: Analyzes surrounding code to detect issues like:
   - Missing colons after network properties
   - Unclosed parentheses and braces
   - Missing quotes around string values
   - Incorrect parameter syntax
3. Fix Hints: Provides specific, actionable steps to resolve errors
4. Quick Reference: Includes syntax reminders in error messages
"""

from dataclasses import dataclass
import difflib
import re
from typing import List, Optional, Tuple

from lark import UnexpectedCharacters, UnexpectedToken

from neural.exceptions import ParserException


@dataclass
class ParserError:
    """Structured representation of a parsing error with full context."""
    message: str
    line: int
    column: int
    context: str
    suggestion: Optional[str] = None
    error_type: Optional[str] = None
    severity: str = "ERROR"
    fix_hint: Optional[str] = None

# Keep for backwards compatibility
class NeuralParserError(ParserException):
    """Custom exception class for Neural parser errors (legacy)."""
    def __init__(self, error: ParserError):
        self.error = error
        super().__init__(
            message=error.message,
            line=error.line,
            column=error.column,
            code_snippet=error.context,
            suggestion=error.suggestion or error.fix_hint
        )

class ErrorHandler:
    """Handles and formats parser errors with helpful context and suggestions.
    
    This class provides intelligent error handling that:
    1. Detects common typos and suggests corrections
    2. Provides code context around errors
    3. Offers specific fix suggestions
    4. Categorizes errors for better user experience
    """

    COMMON_MISTAKES = {
        "Dense": ["dense", "Dence", "DNse", "Dens", "desnse", "Desne"],
        "Conv2D": ["conv2d", "Conv2d", "conv2D", "Con2D", "Conv2", "Cov2D", "Convo2D"],
        "Conv1D": ["conv1d", "Conv1d", "conv1D", "Con1D", "Cov1D"],
        "Conv3D": ["conv3d", "Conv3d", "conv3D", "Con3D", "Cov3D"],
        "Input": ["input", "input_layer", "Iput", "Inpt", "Inp"],
        "Output": ["output", "output_layer", "Ouput", "Outpt", "Outp"],
        "MaxPooling2D": ["maxpooling2d", "MaxPooling2d", "max_pooling2d", "MaxPool2D", "Maxpool2d", "MaxPooling2d"],
        "MaxPooling1D": ["maxpooling1d", "MaxPool1D", "max_pooling1d"],
        "Flatten": ["flatten", "Flaten", "Flatn", "Flatenn"],
        "Dropout": ["dropout", "DropOut", "drop_out", "Droppout", "Dropuot"],
        "LSTM": ["lstm", "Lstm", "LTSM"],
        "GRU": ["gru", "Gru", "GRu"],
        "BatchNormalization": ["batchnormalization", "BatchNorm", "batch_normalization", "batchnorm"],
        "activation": ["activate", "activations", "activ", "actvation", "actiavtion"],
        "filters": ["filter", "filers", "filtes", "filrers", "fliters"],
        "kernel_size": ["kernel_size", "kernal_size", "kernelSize", "kernel", "kernal", "kernel_sie"],
        "units": ["unit", "unites", "unts", "uints"],
        "pool_size": ["poolsize", "poolSize", "pool_size", "poolsie", "pool_szie"],
        "layers": ["layer", "layes", "lyaers", "layres"],
        "input": ["inpt", "inp", "inut"],
        "loss": ["los", "lsos", "losse"],
        "optimizer": ["optimiser", "optmizer", "optimzer", "optimier"],
        "epochs": ["epoch", "epocs", "epohs", "epcohs"],
        "batch_size": ["batchsize", "batch_szie", "batchSize", "batch_sie"],
    }

    VALID_LAYERS = [
        "Dense", "Conv2D", "Conv1D", "Conv3D", "MaxPooling2D", "MaxPooling1D", "MaxPooling3D",
        "Flatten", "Dropout", "Input", "Output", "LSTM", "GRU", "BatchNormalization",
        "GlobalAveragePooling2D", "GlobalAveragePooling1D", "ResidualConnection",
        "Concatenate", "Add", "Multiply", "Average", "Maximum", "Activation",
        "Reshape", "Permute", "ZeroPadding2D", "Cropping2D", "UpSampling2D"
    ]

    VALID_PARAMS = [
        "filters", "kernel_size", "units", "activation", "pool_size", "strides",
        "padding", "rate", "dropout", "input_shape", "output_shape", "use_bias",
        "kernel_initializer", "bias_initializer", "momentum", "epsilon", "axis",
        "return_sequences", "stateful", "data_format"
    ]

    VALID_ACTIVATIONS = [
        "relu", "sigmoid", "tanh", "softmax", "softplus", "softsign", "selu",
        "elu", "exponential", "leaky_relu", "swish", "gelu", "linear"
    ]

    VALID_NETWORK_PROPERTIES = ["input", "layers", "loss", "optimizer", "train", "metrics"]

    @staticmethod
    def get_line_context(code: str, line_no: int, context_lines: int = 3, column: int = None) -> str:
        """Get the surrounding lines of code for context with error indicator."""
        lines = code.splitlines()
        start = max(0, line_no - context_lines)
        end = min(len(lines), line_no + context_lines + 1)

        context = []
        for i in range(start, end):
            line_content = lines[i] if i < len(lines) else ""
            prefix = ">>>" if i == line_no else "   "
            line_num = f"{i+1:4d}"
            context.append(f"{prefix} {line_num} | {line_content}")

            if i == line_no and column is not None:
                arrow = " " * (column + 11) + "^" + "~" * min(5, max(0, len(line_content) - column - 1))
                context.append(f"           {arrow}")

        return "\n".join(context)

    @staticmethod
    def suggest_correction(token: str, context: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Suggest corrections for common mistakes."""
        token_lower = token.lower()

        for correct, mistakes in ErrorHandler.COMMON_MISTAKES.items():
            if token_lower in [m.lower() for m in mistakes]:
                fix_hint = f"Replace '{token}' with '{correct}'"
                return correct, fix_hint

        layer_matches = difflib.get_close_matches(token, ErrorHandler.VALID_LAYERS, n=1, cutoff=0.6)
        if layer_matches:
            fix_hint = f"Did you mean the layer type '{layer_matches[0]}'?"
            return layer_matches[0], fix_hint

        param_matches = difflib.get_close_matches(token, ErrorHandler.VALID_PARAMS, n=1, cutoff=0.7)
        if param_matches:
            fix_hint = f"Did you mean the parameter '{param_matches[0]}'?"
            return param_matches[0], fix_hint

        activation_matches = difflib.get_close_matches(token, ErrorHandler.VALID_ACTIVATIONS, n=1, cutoff=0.7)
        if activation_matches:
            fix_hint = f"Did you mean the activation '{activation_matches[0]}'?"
            return activation_matches[0], fix_hint

        all_valid = ErrorHandler.VALID_LAYERS + ErrorHandler.VALID_PARAMS
        matches = difflib.get_close_matches(token, all_valid, n=1, cutoff=0.5)
        if matches:
            fix_hint = f"Did you mean '{matches[0]}'?"
            return matches[0], fix_hint

        return None, None

    @staticmethod
    def detect_common_issues(code: str, line_no: int, column: int, token: str = None) -> Optional[str]:
        """Detect common issues and provide specific fix hints."""
        lines = code.splitlines()
        if line_no >= len(lines):
            return None

        line = lines[line_no]

        if token and re.match(r'^\w+$', token):
            if token.lower() in [prop.lower() for prop in ErrorHandler.VALID_NETWORK_PROPERTIES]:
                return f"Add a colon (:) after '{token}' to define the network property"

        if re.search(r'\b(layers|input|loss|optimizer|train|metrics)\s*\n', line):
            return "Missing colon (:) after network property. Add ':' after the property name"

        if re.search(r'\b(Conv2D|Dense|Flatten|MaxPooling2D|Dropout|LSTM|GRU)\s*$', line):
            return "Missing opening parenthesis '(' after layer name"

        open_parens = line.count('(')
        close_parens = line.count(')')
        if open_parens > close_parens:
            return f"Missing {open_parens - close_parens} closing parenthesis(es) ')'"
        elif close_parens > open_parens:
            return f"Extra {close_parens - open_parens} closing parenthesis(es) ')' - remove them"

        if re.search(r'activation\s*=\s*[a-zA-Z_]\w*\b(?!["\'])', line):
            return 'String values must be in quotes. Use: activation="relu"'

        if '{' in line and '}' not in line and line_no < len(lines) - 1:
            brace_count = sum(
                line_text.count('{') - line_text.count('}')
                for line_text in lines[:line_no + 1]
            )
            if brace_count > 0:
                return f"Missing {brace_count} closing brace(s) '}}' for the block"

        if re.search(r'=\s*\(\s*[^)]*$', line):
            return "Incomplete tuple definition - ensure all tuples are closed with ')'"

        if re.search(r'kernel_size\s*=\s*\d+\s*,\s*\d+', line):
            return "kernel_size should be a tuple. Use: kernel_size=(3,3)"

        if re.search(r':\s*\(\s*\)', line):
            return "Empty tuple detected. Provide at least one dimension for input shape"

        return None

    @staticmethod
    def suggest_fix_for_expected_tokens(expected: List[str], line: str) -> Optional[str]:
        """Provide actionable fix suggestions based on expected tokens."""
        if not expected:
            return None

        if 'COLON' in expected or ':' in expected:
            return "Add a colon ':' here"

        if 'LPAREN' in expected or '(' in expected:
            return "Add an opening parenthesis '(' here"

        if 'RPAREN' in expected or ')' in expected:
            return "Add a closing parenthesis ')' here"

        if 'RBRACE' in expected or '}' in expected:
            return "Add a closing brace '}' here to close the block"

        if 'COMMA' in expected or ',' in expected:
            return "Add a comma ',' to separate parameters"

        if 'EQUALS' in expected or '=' in expected:
            return "Use '=' to assign parameter values. Example: units=128"

        if 'STRING' in expected:
            return 'Wrap the value in quotes. Example: activation="relu"'

        if 'NUMBER' in expected or 'INT' in expected:
            return "Expected a numeric value here"

        if any('LAYER' in e for e in expected):
            return f"Expected a layer definition. Valid layers: {', '.join(ErrorHandler.VALID_LAYERS[:5])}"

        return None

    @classmethod
    def handle_unexpected_token(cls, error: UnexpectedToken, code: str) -> ParserError:
        """Handle unexpected token errors with enhanced context and suggestions."""
        line_no = error.line - 1
        context = cls.get_line_context(code, line_no, column=error.column - 1)

        token_str = str(error.token)
        suggestion, fix_hint = cls.suggest_correction(token_str, context)

        common_issue = cls.detect_common_issues(code, line_no, error.column - 1, token_str)

        msg = f"Unexpected token '{token_str}' at line {error.line}, column {error.column}"

        if error.expected:
            expected_list = sorted(set(str(e) for e in error.expected))[:5]
            msg += f"\n   Expected one of: {', '.join(expected_list)}"

            fix_from_expected = cls.suggest_fix_for_expected_tokens(expected_list, code.splitlines()[line_no] if line_no < len(code.splitlines()) else "")
            if fix_from_expected and not fix_hint:
                fix_hint = fix_from_expected

        if suggestion:
            msg += f"\n\n💡 Suggestion: Did you mean '{suggestion}'?"

        if fix_hint:
            msg += f"\n🔧 Fix: {fix_hint}"

        if common_issue:
            msg += f"\n⚠️  Common Issue: {common_issue}"

        msg += "\n\n📚 Quick Reference:"
        msg += "\n   - Layer syntax: LayerName(param1=value1, param2=value2)"
        msg += "\n   - Network properties: input:, layers:, loss:, optimizer:"
        msg += "\n   - Strings must be quoted: activation=\"relu\""

        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context,
            suggestion=suggestion,
            error_type="syntax",
            severity="ERROR",
            fix_hint=fix_hint or common_issue
        )

    @classmethod
    def handle_unexpected_char(cls, error: UnexpectedCharacters, code: str) -> ParserError:
        """Handle unexpected character errors with enhanced context."""
        line_no = error.line - 1
        context = cls.get_line_context(code, line_no, column=error.column - 1)

        common_issue = cls.detect_common_issues(code, line_no, error.column - 1)

        char_repr = repr(error.char) if error.char else "''"
        msg = f"Unexpected character {char_repr} at line {error.line}, column {error.column}"

        special_char_hints = {
            '@': "The '@' symbol is used for decorators/annotations. Check if this is intended.",
            '$': "The '$' symbol is not valid in Neural DSL syntax.",
            '%': "Use proper parameter syntax: name=value",
            '&': "Use 'and' keyword instead of '&' symbol",
            '|': "Use 'or' keyword instead of '|' symbol",
            '!': "Use 'not' keyword instead of '!' symbol",
        }

        if error.char in special_char_hints:
            msg += f"\n⚠️  {special_char_hints[error.char]}"

        if hasattr(error, 'allowed') and error.allowed:
            allowed_str = ", ".join(sorted(error.allowed)[:5])
            msg += f"\n   Expected one of: {allowed_str}"

        if common_issue:
            msg += f"\n🔧 Fix: {common_issue}"

        return ParserError(
            message=msg,
            line=error.line,
            column=error.column,
            context=context,
            error_type="syntax",
            severity="ERROR",
            fix_hint=common_issue
        )

    @classmethod
    def handle_shape_error(cls, shape_error: Exception, code: str, line_no: int, layer_name: str = None) -> ParserError:
        """Handle shape propagation errors with enhanced context and suggestions."""
        context = cls.get_line_context(code, line_no)

        error_msg = str(shape_error)

        fix_hint = None
        if "dimension" in error_msg.lower() or "shape" in error_msg.lower():
            fix_hint = "Check that input/output shapes match between layers. Use 'neural visualize' to see shape flow"
        elif "mismatch" in error_msg.lower():
            fix_hint = "Shape mismatch detected. Verify layer parameters (filters, kernel_size, etc.) are compatible"

        if layer_name:
            error_msg = f"Shape error in {layer_name}: {error_msg}"

        return ParserError(
            message=error_msg,
            line=line_no + 1,
            column=0,
            context=context,
            error_type="shape",
            severity="ERROR",
            fix_hint=fix_hint
        )

    @classmethod
    def format_error(cls, error: ParserError) -> str:
        """Format a ParserError into a user-friendly string.
        
        Example output:
        ======================================================================
        ERROR: SYNTAX ERROR
        ======================================================================
        
        Unexpected token 'Desnse' at line 5, column 7
           Expected one of: Dense, Conv2D, Flatten
        
        💡 Suggestion: Did you mean 'Dense'?
        🔧 Fix: Replace 'Desnse' with 'Dense'
        ⚠️  Common Issue: Missing opening parenthesis '(' after layer name
        
        📚 Quick Reference:
           - Layer syntax: LayerName(param1=value1, param2=value2)
           - Network properties: input:, layers:, loss:, optimizer:
           - Strings must be quoted: activation="relu"
        
        📍 Location: Line 5, Column 7
        📄 Context:
            2  | network MyModel {
            3  |   input: (28, 28, 1)
            4  |   layers:
        >>> 5  |     Desnse(units=128)
                      ^~~~~~
            6  |     Output(10)
            7  | }
        
        ======================================================================
        """
        lines = [
            f"\n{'='*70}",
            f"{error.severity}: {error.error_type.upper() if error.error_type else 'PARSER'} ERROR",
            f"{'='*70}",
            f"\n{error.message}",
            f"\n📍 Location: Line {error.line}, Column {error.column}",
            "\n📄 Context:",
            error.context,
        ]

        if error.suggestion:
            lines.append(f"\n💡 Suggestion: {error.suggestion}")

        if error.fix_hint:
            lines.append(f"\n🔧 Fix Hint: {error.fix_hint}")

        lines.append(f"\n{'='*70}\n")

        return "\n".join(lines)
