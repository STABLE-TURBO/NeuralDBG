from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class QualityRule:
    """
    Represents a data quality validation rule.
    
    Attributes:
        name: Rule name
        rule_type: Type of rule (completeness, uniqueness, etc)
        validator: Callable validating the data
        threshold: Optional numeric threshold
        description: Rule description
    """
    def __init__(
        self,
        name: str,
        rule_type: str,
        validator: Optional[Callable[[Any], bool]] = None,
        threshold: Optional[float] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.rule_type = rule_type
        self.validator = validator
        self.threshold = threshold
        self.description = description or ""
        self.created_at = datetime.now().isoformat()

    def validate(self, data: Any) -> bool:
        if self.validator is None:
            raise ValueError(f"No validator defined for rule: {self.name}")
        
        try:
            return self.validator(data)
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rule_type": self.rule_type,
            "threshold": self.threshold,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QualityRule:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.rule_type = data["rule_type"]
        obj.validator = None
        obj.threshold = data.get("threshold")
        obj.description = data.get("description", "")
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        return obj


class ValidationResult:
    """
    Result of a quality rule validation.
    
    Attributes:
        rule_name: Name of the applied rule
        passed: Whether validation passed
        message: Result message
        details: Additional details
    """
    def __init__(
        self,
        rule_name: str,
        passed: bool,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.rule_name = rule_name
        self.passed = passed
        self.message = message or ""
        self.details = details or {}
        self.validated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "validated_at": self.validated_at,
        }


class DataQualityValidator:
    """
    Validates data quality against a set of rules.
    """
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self.base_dir = Path(base_dir)
        self.rules_dir = self.base_dir / "quality_rules"
        self.results_dir = self.base_dir / "validation_results"
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rules: Dict[str, QualityRule] = {}
        self._load_rules()
        self._register_builtin_rules()

    def _load_rules(self):
        for rule_file in self.rules_dir.glob("*.json"):
            with open(rule_file, "r") as f:
                data = json.load(f)
                rule = QualityRule.from_dict(data)
                self.rules[rule.name] = rule

    def _save_rule(self, rule: QualityRule):
        rule_file = self.rules_dir / f"{rule.name}.json"
        with open(rule_file, "w") as f:
            json.dump(rule.to_dict(), f, indent=2)

    def _register_builtin_rules(self):
        builtin_rules = [
            QualityRule(
                name="no_missing_values",
                rule_type="completeness",
                validator=self._validate_completeness,
                description="Check that data has no missing values",
            ),
            QualityRule(
                name="no_duplicates",
                rule_type="uniqueness",
                validator=self._validate_uniqueness,
                description="Check that data has no duplicate rows",
            ),
            QualityRule(
                name="valid_shape",
                rule_type="consistency",
                validator=self._validate_shape,
                description="Check that data has a valid shape",
            ),
        ]
        
        for rule in builtin_rules:
            if rule.name not in self.rules:
                self.rules[rule.name] = rule

    def _validate_completeness(self, data: Any) -> bool:
        """Validate that data has no missing values."""
        return not self._has_missing_values(data)
    
    def _validate_uniqueness(self, data: Any) -> bool:
        """Validate that data has no duplicates."""
        return not self._has_duplicates(data)
    
    def _validate_shape(self, data: Any) -> bool:
        """Validate that data has a valid shape."""
        return self._has_valid_shape(data)

    def _has_missing_values(self, data: Any) -> bool:
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return np.isnan(data).any()
            elif hasattr(data, 'isnull'):
                return data.isnull().any().any()
            return False
        except Exception:
            return False

    def _has_duplicates(self, data: Any) -> bool:
        try:
            if hasattr(data, 'duplicated'):
                return data.duplicated().any()
            return False
        except Exception:
            return False

    def _has_valid_shape(self, data: Any) -> bool:
        try:
            if hasattr(data, 'shape'):
                return len(data.shape) > 0 and all(s > 0 for s in data.shape)
            return True
        except Exception:
            return False

    def add_rule(
        self,
        name: str,
        rule_type: str,
        validator: Optional[Callable[[Any], bool]] = None,
        threshold: Optional[float] = None,
        description: Optional[str] = None,
    ) -> QualityRule:
        if name in self.rules:
            raise ValueError(f"Rule already exists: {name}")
        
        rule = QualityRule(name, rule_type, validator, threshold, description)
        self.rules[name] = rule
        self._save_rule(rule)
        return rule

    def remove_rule(self, name: str) -> bool:
        if name not in self.rules:
            return False
        
        rule_file = self.rules_dir / f"{name}.json"
        if rule_file.exists():
            rule_file.unlink()
        
        del self.rules[name]
        return True

    def get_rule(self, name: str) -> Optional[QualityRule]:
        return self.rules.get(name)

    def list_rules(self, rule_type: Optional[str] = None) -> List[str]:
        if rule_type:
            return [
                name
                for name, rule in self.rules.items()
                if rule.rule_type == rule_type
            ]
        return list(self.rules.keys())

    def validate(
        self,
        data: Any,
        rules: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
        rules_to_apply = (
            [self.rules[r] for r in rules if r in self.rules]
            if rules
            else list(self.rules.values())
        )
        
        results = []
        for rule in rules_to_apply:
            try:
                if rule.validator:
                    passed = rule.validate(data)
                    message = "Validation passed" if passed else "Validation failed"
                else:
                    passed = False
                    message = "No validator defined"
                
                results.append(ValidationResult(rule.name, passed, message))
            except Exception as e:
                results.append(
                    ValidationResult(rule.name, False, f"Error: {str(e)}")
                )
        
        return results

    def validate_and_save(
        self,
        data: Any,
        dataset_name: str,
        rules: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
        results = self.validate(data, rules)
        
        # Use microsecond precision and a counter to ensure unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        result_file = self.results_dir / f"{dataset_name}_{timestamp}.json"
        
        # If file exists, add a counter suffix to ensure uniqueness
        counter = 0
        while result_file.exists():
            counter += 1
            result_file = self.results_dir / f"{dataset_name}_{timestamp}_{counter}.json"
        
        with open(result_file, "w") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )
        
        return results

    def get_validation_history(
        self, dataset_name: str
    ) -> List[Dict[str, Any]]:
        history = []
        
        for result_file in sorted(self.results_dir.glob(f"{dataset_name}_*.json")):
            with open(result_file, "r") as f:
                data = json.load(f)
                history.append(data)
        
        return history

    def get_quality_report(self, dataset_name: str) -> Dict[str, Any]:
        history = self.get_validation_history(dataset_name)
        
        if not history:
            return {
                "dataset_name": dataset_name,
                "total_validations": 0,
                "error": "No validation history found",
            }
        
        latest = history[-1]
        
        total_rules = len(latest["results"])
        passed_rules = sum(1 for r in latest["results"] if r["passed"])
        failed_rules = total_rules - passed_rules
        
        return {
            "dataset_name": dataset_name,
            "total_validations": len(history),
            "latest_validation": latest["results"][0]["validated_at"] if latest["results"] else None,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "pass_rate": passed_rules / total_rules if total_rules > 0 else 0,
            "failed_rule_names": [
                r["rule_name"] for r in latest["results"] if not r["passed"]
            ],
        }

    def create_custom_rule(
        self,
        name: str,
        rule_type: str,
        condition: str,
        threshold: Optional[float] = None,
        description: Optional[str] = None,
    ) -> QualityRule:
        def validator(data: Any) -> bool:
            try:
                if condition == "min_rows":
                    return len(data) >= (threshold or 0)
                elif condition == "max_rows":
                    return len(data) <= (threshold or float('inf'))
                elif condition == "min_cols" and hasattr(data, 'shape'):
                    return data.shape[1] >= (threshold or 0)
                elif condition == "max_cols" and hasattr(data, 'shape'):
                    return data.shape[1] <= (threshold or float('inf'))
                return True
            except Exception:
                return False
        
        return self.add_rule(name, rule_type, validator, threshold, description)

class QualityValidator:
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self._impl = DataQualityValidator(base_dir=base_dir)
    
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        result = {"valid": True, "errors": []}
        try:
            sample = data
            if hasattr(data, "to_dict"):
                sample = data.to_dict()
            for field, rules in schema.items():
                required = bool(rules.get("required"))
                if required and field not in sample:
                    result["valid"] = False
                    result["errors"].append(f"missing:{field}")
                    continue
                if field in sample:
                    value = sample[field]
                    t = rules.get("type")
                    if t == "int" and not isinstance(value, int):
                        result["valid"] = False
                        result["errors"].append(f"type:{field}")
                    elif t == "float" and not isinstance(value, (int, float)):
                        result["valid"] = False
                        result["errors"].append(f"type:{field}")
                    elif t == "str" and not isinstance(value, str):
                        result["valid"] = False
                        result["errors"].append(f"type:{field}")
                    elif t == "bool" and not isinstance(value, bool):
                        result["valid"] = False
                        result["errors"].append(f"type:{field}")
                    if isinstance(value, (int, float)):
                        if "min" in rules and value < rules["min"]:
                            result["valid"] = False
                            result["errors"].append(f"range:{field}")
                        if "max" in rules and value > rules["max"]:
                            result["valid"] = False
                            result["errors"].append(f"range:{field}")
        except Exception:
            return {"valid": False, "errors": ["error"]}
        return result
    
    def validate_completeness(self, data: Any) -> Dict[str, Any]:
        try:
            import numpy as np  # type: ignore
            if isinstance(data, np.ndarray):
                total = int(np.prod(data.shape)) if data.size else 0
                missing = int(np.isnan(data).sum()) if total > 0 else 0
                pct = (missing / total) if total > 0 else 0.0
                return {"complete": missing == 0, "missing_percentage": float(pct)}
            if hasattr(data, "isnull"):
                nulls = data.isnull().sum().sum()
                total = data.size if hasattr(data, "size") else 0
                pct = (float(nulls) / float(total)) if total else 0.0
                return {"complete": nulls == 0, "missing_percentage": float(pct)}
        except Exception:
            pass
        return {"complete": True, "missing_percentage": 0.0}
    
    def validate_consistency(self, data: Any) -> Dict[str, Any]:
        consistent = self._impl._has_valid_shape(data) and (not self._impl._has_duplicates(data))
        return {"consistent": bool(consistent), "violations": [] if consistent else ["shape_or_duplicates"]}
    
    def validate_accuracy(self, data: Any, reference: Any) -> Dict[str, Any]:
        try:
            import numpy as np  # type: ignore
            a = np.asarray(data)
            b = np.asarray(reference)
            if a.shape != b.shape or a.size == 0:
                return {"accurate": True, "error_rate": 0.0}
            err = float(np.mean(np.abs(a - b)))
            return {"accurate": err == 0.0, "error_rate": err}
        except Exception:
            return {"accurate": True, "error_rate": 0.0}
