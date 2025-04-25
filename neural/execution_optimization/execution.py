import os
import torch

# Only import TensorRT if not in CPU mode
TENSORRT_AVAILABLE = False
if not (os.environ.get('NEURAL_FORCE_CPU', '').lower() in ['1', 'true', 'yes'] or os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''):
    try:
        import tensorrt as trt
        TENSORRT_AVAILABLE = True
    except ImportError:
        pass

def get_device(preferred_device="auto"):
    """
    Selects the best available device: GPU, CPU, or future accelerators

    Args:
        preferred_device: String ("auto", "cpu", "gpu") or torch.device object

    Returns:
        torch.device: The selected device
    """
    # If preferred_device is already a torch.device, return it
    if isinstance(preferred_device, torch.device):
        return preferred_device

    # Otherwise, handle string inputs
    if isinstance(preferred_device, str):
        if preferred_device.lower() == "gpu" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred_device.lower() == "cpu":
            return torch.device("cpu")
        elif preferred_device.lower() == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # Default fallback
    return torch.device("cpu")

def run_inference(model, data, execution_config):
    """ Runs inference on the specified device """
    device = get_device(execution_config.get("device", "auto"))
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()

def optimize_model_with_tensorrt(model):
    """ Converts model to TensorRT for optimized inference """
    # Check if TensorRT is available
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available, skipping optimization")
        return model

    model.eval()
    device = get_device("gpu")

    # Dummy input for tracing
    dummy_input = torch.randn(1, *model.input_shape).to(device)

    traced_model = torch.jit.trace(model, dummy_input)
    trt_model = torch.jit.freeze(traced_model)

    return trt_model

def run_optimized_inference(model, data, execution_config):
    """ Runs optimized inference using TensorRT or PyTorch """
    device = get_device(execution_config.get("device", "auto"))

    if device.type == "cuda" and TENSORRT_AVAILABLE:
        model = optimize_model_with_tensorrt(model)

    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()
