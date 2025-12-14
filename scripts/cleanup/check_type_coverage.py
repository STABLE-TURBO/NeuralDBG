#!/usr/bin/env python3
"""
Check type hint coverage in Neural DSL codebase.

Analyzes Python files to determine what percentage of functions
and methods have type hints for parameters and return values.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ModuleStats:
    """Statistics for a single module."""
    module_path: str
    total_functions: int
    typed_functions: int
    partially_typed: int
    untyped_functions: int
    
    @property
    def coverage_percent(self) -> float:
        if self.total_functions == 0:
            return 100.0
        return (self.typed_functions / self.total_functions) * 100


class TypeCoverageAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze type hint coverage."""
    
    def __init__(self):
        self.total = 0
        self.fully_typed = 0
        self.partially_typed = 0
        self.untyped = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check type hints on function definitions."""
        self.total += 1
        
        # Skip special methods and private functions for now
        if node.name.startswith('_') and node.name != '__init__':
            self.generic_visit(node)
            return
        
        # Check parameter annotations
        params_typed = all(
            arg.annotation is not None 
            for arg in node.args.args 
            if arg.arg != 'self' and arg.arg != 'cls'
        )
        
        # Check return annotation
        return_typed = node.returns is not None
        
        if params_typed and return_typed:
            self.fully_typed += 1
        elif params_typed or return_typed:
            self.partially_typed += 1
        else:
            self.untyped += 1
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check type hints on async function definitions."""
        self.visit_FunctionDef(node)


def analyze_file(file_path: Path) -> Tuple[int, int, int, int]:
    """Analyze a single Python file for type coverage.
    
    Returns:
        (total, fully_typed, partially_typed, untyped)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        analyzer = TypeCoverageAnalyzer()
        analyzer.visit(tree)
        
        return (
            analyzer.total,
            analyzer.fully_typed,
            analyzer.partially_typed,
            analyzer.untyped
        )
    except Exception as e:
        print(f"  Error analyzing {file_path}: {e}")
        return (0, 0, 0, 0)


def analyze_directory(dir_path: Path) -> ModuleStats:
    """Analyze all Python files in a directory."""
    total = 0
    fully_typed = 0
    partially_typed = 0
    untyped = 0
    
    for py_file in dir_path.rglob("*.py"):
        # Skip __pycache__ and test files
        if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
            continue
        
        t, f, p, u = analyze_file(py_file)
        total += t
        fully_typed += f
        partially_typed += p
        untyped += u
    
    return ModuleStats(
        module_path=str(dir_path.relative_to(Path(__file__).parent.parent.parent)),
        total_functions=total,
        typed_functions=fully_typed,
        partially_typed=partially_typed,
        untyped_functions=untyped
    )


def main():
    """Main function to run type coverage analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check type hint coverage in Neural DSL"
    )
    parser.add_argument(
        "--module",
        help="Specific module to check (e.g., neural/code_generation)",
        default=None
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-file breakdown"
    )
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent.parent
    neural_dir = root / "neural"
    
    print("Type Hint Coverage Analysis")
    print("=" * 70)
    print()
    
    # Priority modules to check
    priority_modules = [
        "code_generation",
        "utils",
        "shape_propagation",
        "parser",
        "cli",
        "dashboard",
    ]
    
    if args.module:
        modules_to_check = [args.module.replace("/", "").replace("\\", "")]
    else:
        modules_to_check = priority_modules
    
    all_stats: List[ModuleStats] = []
    
    for module_name in modules_to_check:
        module_path = neural_dir / module_name
        
        if not module_path.exists():
            print(f"⚠ Module not found: {module_name}")
            continue
        
        stats = analyze_directory(module_path)
        all_stats.append(stats)
        
        # Determine status emoji
        coverage = stats.coverage_percent
        if coverage >= 90:
            status = "✅"
        elif coverage >= 70:
            status = "🟡"
        else:
            status = "❌"
        
        print(f"{status} {module_name}/")
        print(f"   Coverage: {coverage:.1f}%")
        print(f"   Fully typed: {stats.typed_functions}/{stats.total_functions}")
        print(f"   Partially typed: {stats.partially_typed}")
        print(f"   Untyped: {stats.untyped_functions}")
        print()
    
    # Overall summary
    if len(all_stats) > 1:
        total_funcs = sum(s.total_functions for s in all_stats)
        total_typed = sum(s.typed_functions for s in all_stats)
        
        if total_funcs > 0:
            overall_coverage = (total_typed / total_funcs) * 100
        else:
            overall_coverage = 100.0
        
        print("=" * 70)
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        print(f"Total Functions: {total_funcs}")
        print(f"Fully Typed: {total_typed}")
        print("=" * 70)
    
    # Recommendations
    print("\nRecommendations:")
    print("-" * 70)
    
    for stats in all_stats:
        if stats.coverage_percent < 90:
            print(f"• {stats.module_path}: Improve to 90%+ coverage")
            print(f"  Need to type {stats.untyped_functions + stats.partially_typed} more functions")
    
    if all(s.coverage_percent >= 90 for s in all_stats):
        print("✅ All checked modules have excellent type coverage!")


if __name__ == "__main__":
    main()
