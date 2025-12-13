"""
Utility functions for the Runner Panel
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_log_line(line: str) -> Dict[str, Any]:
    """
    Parse a log line to extract information and format it
    
    Args:
        line: Raw log line from process output
        
    Returns:
        Dictionary with parsed information
    """
    result = {
        "raw": line,
        "type": "info",
        "message": line.strip(),
        "metrics": {}
    }
    
    line_lower = line.lower()
    
    # Determine log type
    if "error" in line_lower or "exception" in line_lower or "failed" in line_lower:
        result["type"] = "error"
    elif "warning" in line_lower or "warn" in line_lower:
        result["type"] = "warning"
    elif "success" in line_lower or "completed" in line_lower or "✓" in line:
        result["type"] = "success"
    elif "[compile]" in line_lower or "[run]" in line_lower:
        result["type"] = "system"
    
    # Extract metrics if present
    metrics = extract_metrics(line)
    if metrics:
        result["metrics"] = metrics
        result["type"] = "metrics"
    
    return result


def extract_metrics(line: str) -> Dict[str, float]:
    """
    Extract training metrics from a log line
    
    Common patterns:
    - "Epoch 1/10 - loss: 0.5234 - accuracy: 0.8912"
    - "loss: 0.5234 accuracy: 0.8912"
    - "Train Loss: 0.5234, Train Acc: 89.12%"
    """
    metrics = {}
    
    # Pattern 1: loss: <value>
    loss_match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    
    # Pattern 2: accuracy: <value> or acc: <value>
    acc_match = re.search(r'acc(?:uracy)?[:\s]+([0-9.]+)', line.lower())
    if acc_match:
        acc_value = float(acc_match.group(1))
        # Convert percentage to decimal if needed
        if acc_value > 1:
            acc_value /= 100
        metrics['accuracy'] = acc_value
    
    # Pattern 3: val_loss: <value>
    val_loss_match = re.search(r'val[_\s]?loss[:\s]+([0-9.]+)', line.lower())
    if val_loss_match:
        metrics['val_loss'] = float(val_loss_match.group(1))
    
    # Pattern 4: val_accuracy: <value>
    val_acc_match = re.search(r'val[_\s]?acc(?:uracy)?[:\s]+([0-9.]+)', line.lower())
    if val_acc_match:
        val_acc_value = float(val_acc_match.group(1))
        if val_acc_value > 1:
            val_acc_value /= 100
        metrics['val_accuracy'] = val_acc_value
    
    # Pattern 5: epoch number
    epoch_match = re.search(r'epoch[:\s]+(\d+)', line.lower())
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    
    return metrics


def format_console_line(parsed_line: Dict[str, Any]) -> str:
    """
    Format a parsed log line for console display with color coding
    
    Args:
        parsed_line: Parsed log line dictionary
        
    Returns:
        Formatted string with ANSI color codes (for future enhancement)
    """
    line_type = parsed_line.get("type", "info")
    message = parsed_line.get("message", "")
    
    # For now, just add prefixes based on type
    # In future, could add ANSI color codes or HTML formatting
    prefixes = {
        "error": "[ERROR] ",
        "warning": "[WARN] ",
        "success": "[SUCCESS] ",
        "system": "",
        "metrics": "[METRICS] ",
        "info": ""
    }
    
    prefix = prefixes.get(line_type, "")
    return prefix + message


def validate_training_config(
    epochs: int,
    batch_size: int,
    validation_split: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate training configuration parameters
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if epochs < 1:
        return False, "Epochs must be at least 1"
    
    if epochs > 1000:
        return False, "Epochs cannot exceed 1000"
    
    if batch_size < 1:
        return False, "Batch size must be at least 1"
    
    if batch_size > 2048:
        return False, "Batch size cannot exceed 2048"
    
    if validation_split < 0 or validation_split >= 1:
        return False, "Validation split must be between 0 and 1"
    
    return True, None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    epochs: int,
    time_per_batch: float = 0.1
) -> str:
    """
    Estimate training time based on configuration
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        time_per_batch: Average time per batch in seconds
        
    Returns:
        Formatted time estimate (e.g., "~5 minutes")
    """
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * time_per_batch
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds / 60)} minutes"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"~{hours}h {minutes}m"


def truncate_output(output: str, max_lines: int = 1000) -> str:
    """
    Truncate output to keep console manageable
    
    Args:
        output: Full output string
        max_lines: Maximum number of lines to keep
        
    Returns:
        Truncated output
    """
    lines = output.split('\n')
    
    if len(lines) <= max_lines:
        return output
    
    # Keep last max_lines lines
    truncated_lines = lines[-max_lines:]
    truncation_notice = f"... (truncated, showing last {max_lines} lines) ...\n\n"
    
    return truncation_notice + '\n'.join(truncated_lines)


def get_backend_display_name(backend: str) -> str:
    """
    Get display name for backend
    
    Args:
        backend: Backend identifier
        
    Returns:
        Display name
    """
    display_names = {
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "onnx": "ONNX"
    }
    return display_names.get(backend.lower(), backend)


def get_dataset_display_name(dataset: str) -> str:
    """
    Get display name for dataset
    
    Args:
        dataset: Dataset identifier
        
    Returns:
        Display name
    """
    return dataset.upper() if dataset.upper() in ["MNIST", "CIFAR10", "CIFAR100"] else dataset
