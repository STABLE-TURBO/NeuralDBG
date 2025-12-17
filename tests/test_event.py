import torch
import pytest
from neuraldbg import Event


class TestEventUnit:
    """Unit tests for individual Event methods."""

    def test_init_sets_attributes(self):
        """Test that __init__ correctly sets all attributes."""
        input_tensor = torch.randn(3, 3, requires_grad=True)
        output_tensor = torch.randn(3, 3, requires_grad=True)

        event = Event(step=1, layer_name='conv1', input_tensor=input_tensor, output_tensor=output_tensor)

        assert event.step == 1
        assert event.layer_name == 'conv1'
        assert torch.equal(event.input, input_tensor)
        assert torch.equal(event.output, output_tensor)
        assert event.gradient is None
        # Check that they are detached and cloned
        assert not event.input.requires_grad
        assert not event.output.requires_grad
        assert event.input is not input_tensor
        assert event.output is not output_tensor

    def test_capture_gradient_with_grad(self):
        """Test capture_gradient when tensor has gradients."""
        tensor = torch.randn(3, 3, requires_grad=True)
        loss = tensor.sum()
        loss.backward()

        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        assert event.gradient is not None
        assert torch.equal(event.gradient, tensor.grad)
        assert not event.gradient.requires_grad
        assert event.gradient is not tensor.grad

    def test_capture_gradient_without_grad(self):
        """Test capture_gradient when tensor has no gradients."""
        tensor = torch.randn(3, 3, requires_grad=False)

        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        assert event.gradient is None


class TestEventIntegration:
    """Integration tests for Event in a simulated neural network context."""

    def test_forward_backward_simulation(self):
        """Test Events capturing forward and backward passes."""
        # Simulate a simple linear layer: y = Wx + b
        input_tensor = torch.randn(10, 5, requires_grad=True)
        weight = torch.randn(3, 5, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)

        output_tensor = torch.matmul(input_tensor, weight.t()) + bias
        loss = output_tensor.sum()
        loss.backward()

        # Create events
        forward_event = Event(step=1, layer_name='linear', input_tensor=input_tensor, output_tensor=output_tensor)
        forward_event.capture_gradient(output_tensor)

        # Check attributes
        assert forward_event.step == 1
        assert forward_event.layer_name == 'linear'
        assert torch.equal(forward_event.input, input_tensor)
        assert torch.equal(forward_event.output, output_tensor)
        assert forward_event.gradient is not None
        assert torch.equal(forward_event.gradient, output_tensor.grad)


class TestEventLogic:
    """Logic tests for edge cases and specific behaviors."""

    def test_tensor_independence(self):
        """Test that Event's tensors are independent of originals."""
        original_input = torch.randn(2, 2, requires_grad=True)
        original_output = torch.randn(2, 2, requires_grad=True)

        event = Event(step=1, layer_name='test', input_tensor=original_input, output_tensor=original_output)

        # Modify originals
        original_input.add_(1)
        original_output.mul_(2)

        # Event's copies should be unchanged
        assert not torch.equal(event.input, original_input)
        assert not torch.equal(event.output, original_output)
        # Original values should be preserved
        assert torch.equal(event.input, original_input - 1)
        assert torch.equal(event.output, original_output / 2)

    def test_gradient_capture_multiple_times(self):
        """Test capturing gradient multiple times updates correctly."""
        tensor = torch.randn(3, requires_grad=True)
        loss = tensor.sum()
        loss.backward()

        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        first_grad = event.gradient.clone()

        # Compute new gradients
        tensor.grad.zero_()
        loss2 = tensor.sum()
        loss2.backward()

        event.capture_gradient(tensor)

        # Should have updated gradient
        assert not torch.equal(event.gradient, first_grad)
        assert torch.equal(event.gradient, tensor.grad)

    def test_capture_gradient_on_different_tensor(self):
        """Test capturing gradient from a different tensor."""
        output_tensor = torch.randn(3, requires_grad=True)
        other_tensor = torch.randn(3, requires_grad=True)

        loss = output_tensor.sum()
        loss.backward()

        event = Event(step=1, layer_name='test', input_tensor=output_tensor, output_tensor=output_tensor)
        event.capture_gradient(other_tensor)  # other_tensor has no grad

        assert event.gradient is None
