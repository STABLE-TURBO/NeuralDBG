"""
NeuralDbg Event Module

This module defines the Event class, which is a core component of the NeuralDbg debugging framework.
The Event class captures and stores information about individual computation steps in a neural network,
allowing for detailed inspection of forward passes, backward passes, and gradient flow.
"""

import torch

class Event:
    """
    Represents a single computation event in a neural network's forward or backward pass.

    An Event captures the state of a neural network layer at a specific step, including
    input/output tensors and gradients. This enables debugging capabilities such as
    tracing tensor transformations, detecting vanishing gradients, and analyzing
    the network's internal state during training.

    Attributes:
        step (int): The sequential step number of this event in the computation trace.
        layer_name (str): The name or identifier of the neural network layer.
        input (torch.Tensor): A detached clone of the input tensor to this layer.
        output (torch.Tensor): A detached clone of the output tensor from this layer.
        gradient (torch.Tensor or None): The gradient tensor captured during backward pass,
                                         initially None until capture_gradient is called.
    """

    def __init__(self, step, layer_name, input_tensor, output_tensor):
        """
        Initialize an Event with the given step information and tensors.

        Args:
            step (int): The step number in the sequence of network computations.
            layer_name (str): Name of the layer this event represents (e.g., 'conv1', 'fc2').
            input_tensor (torch.Tensor): The input tensor fed into the layer.
            output_tensor (torch.Tensor): The output tensor produced by the layer.

        Note:
            Input and output tensors are detached and cloned to create independent copies
            that won't interfere with the original computation graph or autograd.
            This allows safe inspection without affecting training.
        """
        # Store the step number for ordering events in the trace
        self.step = step

        # Store the layer identifier for easy reference
        self.layer_name = layer_name

        # Create detached clones of input and output tensors
        # .detach() removes them from the computation graph
        # .clone() creates independent copies to prevent unintended modifications
        self.input = input_tensor.detach().clone()
        self.output = output_tensor.detach().clone()

        # Initialize gradient as None - will be populated during backward pass
        self.gradient = None  # will be filled after backward

    def capture_gradient(self, tensor):
        """
        Capture the gradient information for this event during the backward pass.

        This method should be called after the backward pass on the tensor to capture
        the computed gradients. Gradients are crucial for understanding how errors
        propagate through the network and for detecting issues like vanishing gradients.

        Args:
            tensor (torch.Tensor): The tensor whose gradient should be captured.
                                  Typically the output tensor of the layer.

        Note:
            Only captures gradients if they exist (tensor.grad is not None).
            Like input/output tensors, gradients are detached and cloned for safe storage.
        """
        # Check if the tensor has gradients computed during backward pass
        if tensor.grad is not None:
            # Detach and clone the gradient to create an independent copy
            # This prevents interference with ongoing computations
            self.gradient = tensor.grad.detach().clone()
