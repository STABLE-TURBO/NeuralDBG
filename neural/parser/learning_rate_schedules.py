"""
Learning rate schedule handlers for Neural DSL.
"""

from typing import Dict, Any
from .validation import validate_numeric, ValidationError

class LearningRateSchedule:
    """Base class for learning rate schedules."""

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize schedule parameters."""
        raise NotImplementedError

class ExponentialDecaySchedule(LearningRateSchedule):
    """Exponential decay learning rate schedule."""

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate exponential decay parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Validated parameters
            
        Raises:
            ValidationError: If validation fails
        """
        validated = {}

        # Validate initial learning rate
        try:
            validated['initial_learning_rate'] = validate_numeric(
                params.get('initial_learning_rate'),
                'initial_learning_rate',
                min_value=0.0
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid initial_learning_rate: {str(e)}")

        # Validate decay steps
        try:
            validated['decay_steps'] = validate_numeric(
                params.get('decay_steps'),
                'decay_steps',
                min_value=1,
                integer_only=True
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid decay_steps: {str(e)}")

        # Validate decay rate
        try:
            validated['decay_rate'] = validate_numeric(
                params.get('decay_rate'),
                'decay_rate',
                min_value=0.0
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid decay_rate: {str(e)}")

        # Copy any additional parameters
        for key, value in params.items():
            if key not in validated:
                validated[key] = value

        return validated
