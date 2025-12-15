#!/usr/bin/env python3
"""
Script to run pytest with coverage and generate TEST_COVERAGE_SUMMARY.md

Usage:
    python generate_test_coverage_summary.py
"""

import subprocess
import sys
import re
import json
from pathlib import Path
from datetime import datetime


def run_pytest_with_coverage():
    """Run pytest with coverage options."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/", "-v",
        "--cov=neural",
        "--cov-report=term",
        "--cov-report=html",
        "--cov-report=json",
        "--tb=short"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    return result


def parse_coverage_json():
    """Parse coverage.json file if it exists."""
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        return None
    
    with open(coverage_file, "r") as f:
        return json.load(f)


def parse_pytest_output(stdout, stderr):
    """Parse pytest output to extract test statistics."""
    combined_output = stdout + "\n" + stderr
    
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "warnings": 0,
        "duration": 0.0
    }
    
    # Look for the test summary line
    # e.g., "===== 357 passed, 132 failed, 11 skipped in 45.67s ====="
    summary_pattern = r"=+\s*(.*?)\s+in\s+([\d.]+)s?\s*=+"
    match = re.search(summary_pattern, combined_output)
    
    if match:
        summary_text = match.group(1)
        stats["duration"] = float(match.group(2))
        
        # Parse individual counts
        patterns = {
            "passed": r"(\d+)\s+passed",
            "failed": r"(\d+)\s+failed",
            "skipped": r"(\d+)\s+skipped",
            "errors": r"(\d+)\s+error",
            "xfailed": r"(\d+)\s+xfailed",
            "xpassed": r"(\d+)\s+xpassed",
            "warnings": r"(\d+)\s+warning"
        }
        
        for key, pattern in patterns.items():
            m = re.search(pattern, summary_text)
            if m:
                stats[key] = int(m.group(1))
        
        # Calculate total
        stats["total"] = sum([
            stats["passed"],
            stats["failed"],
            stats["skipped"],
            stats["errors"],
            stats["xfailed"],
            stats["xpassed"]
        ])
    
    # Alternative: look for collected tests
    collected_pattern = r"collected\s+(\d+)\s+item"
    match = re.search(collected_pattern, combined_output)
    if match and stats["total"] == 0:
        stats["total"] = int(match.group(1))
    
    return stats


def extract_module_coverage(coverage_data):
    """Extract per-module coverage from coverage data."""
    if not coverage_data:
        return {}
    
    module_coverage = {}
    files = coverage_data.get("files", {})
    
    for filepath, data in files.items():
        # Group by top-level module
        if filepath.startswith("neural/"):
            parts = filepath.split("/")
            if len(parts) >= 2:
                module = parts[1]
                
                if module not in module_coverage:
                    module_coverage[module] = {
                        "lines": 0,
                        "covered": 0,
                        "missing": 0,
                        "files": 0
                    }
                
                summary = data.get("summary", {})
                module_coverage[module]["lines"] += summary.get("num_statements", 0)
                module_coverage[module]["covered"] += summary.get("covered_lines", 0)
                module_coverage[module]["missing"] += summary.get("missing_lines", 0)
                module_coverage[module]["files"] += 1
    
    # Calculate percentages
    for module, data in module_coverage.items():
        if data["lines"] > 0:
            data["percent"] = (data["covered"] / data["lines"]) * 100
        else:
            data["percent"] = 0.0
    
    return module_coverage


def generate_summary_markdown(stats, coverage_data, stdout, stderr):
    """Generate the TEST_COVERAGE_SUMMARY.md content."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate percentages
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    skipped = stats["skipped"]
    
    pass_rate = (passed / total * 100) if total > 0 else 0
    fail_rate = (failed / total * 100) if total > 0 else 0
    skip_rate = (skipped / total * 100) if total > 0 else 0
    
    # Get overall coverage
    overall_coverage = 0.0
    if coverage_data:
        totals = coverage_data.get("totals", {})
        num_statements = totals.get("num_statements", 0)
        covered_lines = totals.get("covered_lines", 0)
        if num_statements > 0:
            overall_coverage = (covered_lines / num_statements) * 100
    
    module_coverage = extract_module_coverage(coverage_data)
    
    # Build markdown content
    md = f"""# Neural DSL - Test Coverage Summary

**Generated:** {timestamp}  
**Command:** `pytest tests/ -v --cov=neural --cov-report=term --cov-report=html`  
**Python Version:** {sys.version.split()[0]}  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | {total} |
| **Passed** | {passed} ({pass_rate:.1f}%) |
| **Failed** | {failed} ({fail_rate:.1f}%) |
| **Skipped** | {skipped} ({skip_rate:.1f}%) |
| **Errors** | {stats['errors']} |
| **Duration** | {stats['duration']:.2f}s |
| **Overall Coverage** | {overall_coverage:.1f}% |

### Status: {"✅ PASSING" if failed == 0 else "⚠️ NEEDS ATTENTION" if pass_rate >= 90 else "❌ FAILING"}

---

## Coverage by Module

"""
    
    if module_coverage:
        md += "| Module | Coverage | Lines | Covered | Missing | Files |\n"
        md += "|--------|----------|-------|---------|---------|-------|\n"
        
        # Sort by coverage percentage
        sorted_modules = sorted(
            module_coverage.items(),
            key=lambda x: x[1]["percent"],
            reverse=True
        )
        
        for module, data in sorted_modules:
            status = "🟢" if data["percent"] >= 80 else "🟡" if data["percent"] >= 60 else "🔴"
            md += f"| {status} {module} | {data['percent']:.1f}% | {data['lines']} | {data['covered']} | {data['missing']} | {data['files']} |\n"
        
        md += "\n**Legend:** 🟢 ≥80% | 🟡 60-79% | 🔴 <60%\n"
    else:
        md += "*Coverage data not available. Ensure pytest-cov is installed.*\n"
    
    md += "\n---\n\n## Test Results Details\n\n"
    
    if stats["passed"] > 0:
        md += f"### ✅ Passing Tests: {passed}/{total}\n\n"
        md += f"Success rate: {pass_rate:.1f}%\n\n"
    
    if stats["failed"] > 0:
        md += f"### ❌ Failed Tests: {failed}/{total}\n\n"
        md += "Failed tests require investigation. See detailed output below or in htmlcov/index.html\n\n"
    
    if stats["skipped"] > 0:
        md += f"### ⏭️ Skipped Tests: {skipped}/{total}\n\n"
        md += "Skipped tests may be marked with @pytest.mark.skip or require optional dependencies.\n\n"
    
    if stats["errors"] > 0:
        md += f"### 🔴 Errors: {stats['errors']}\n\n"
        md += "Test collection or setup errors. These prevent tests from running.\n\n"
    
    md += "---\n\n## Coverage Report\n\n"
    md += "Detailed coverage report available at: `htmlcov/index.html`\n\n"
    md += "Open in browser:\n"
    md += "- **Windows:** `start htmlcov/index.html`\n"
    md += "- **macOS:** `open htmlcov/index.html`\n"
    md += "- **Linux:** `xdg-open htmlcov/index.html`\n\n"
    
    md += "---\n\n## Improvements Since Last Run\n\n"
    
    # Try to read previous summary if it exists
    prev_summary = Path("TEST_COVERAGE_SUMMARY.md")
    if prev_summary.exists():
        # Extract previous stats
        prev_content = prev_summary.read_text()
        prev_passed_match = re.search(r"\*\*Passed\*\* \| (\d+)", prev_content)
        prev_coverage_match = re.search(r"\*\*Overall Coverage\*\* \| ([\d.]+)%", prev_content)
        
        if prev_passed_match and prev_coverage_match:
            prev_passed = int(prev_passed_match.group(1))
            prev_coverage = float(prev_coverage_match.group(1))
            
            passed_delta = passed - prev_passed
            coverage_delta = overall_coverage - prev_coverage
            
            md += f"- Tests: {passed} (Δ {passed_delta:+d})\n"
            md += f"- Coverage: {overall_coverage:.1f}% (Δ {coverage_delta:+.1f}%)\n\n"
        else:
            md += "*First test run - no previous data for comparison.*\n\n"
    else:
        md += "*First test run - no previous data for comparison.*\n\n"
    
    md += "---\n\n## Recommendations\n\n"
    
    if pass_rate < 100:
        md += "### Priority Actions\n\n"
        
        if stats["errors"] > 0:
            md += "1. **Fix test collection errors** - These prevent tests from running\n"
        
        if fail_rate > 10:
            md += "2. **Investigate failing tests** - High failure rate needs attention\n"
        
        if overall_coverage < 80:
            md += "3. **Improve test coverage** - Add tests for uncovered code paths\n"
        
        md += "\n"
    else:
        md += "✅ All tests passing! Great work!\n\n"
        if overall_coverage < 90:
            md += "Consider adding more tests to improve coverage above 90%.\n\n"
    
    md += "### Next Steps\n\n"
    md += "1. Review failing tests and fix bugs or update tests\n"
    md += "2. Check coverage report for untested code paths\n"
    md += "3. Add tests for new features or bug fixes\n"
    md += "4. Re-run this script to track progress\n\n"
    
    md += "---\n\n## Full Output\n\n"
    md += "<details>\n<summary>Click to expand pytest output</summary>\n\n"
    md += "```\n"
    md += stdout[-10000:] if len(stdout) > 10000 else stdout  # Last 10k chars
    md += "```\n\n"
    md += "</details>\n\n"
    
    if stderr and len(stderr.strip()) > 0:
        md += "<details>\n<summary>Click to expand stderr output</summary>\n\n"
        md += "```\n"
        md += stderr[-5000:] if len(stderr) > 5000 else stderr  # Last 5k chars
        md += "```\n\n"
        md += "</details>\n\n"
    
    md += "---\n\n"
    md += "*This summary was automatically generated by `generate_test_coverage_summary.py`*\n"
    
    return md


def main():
    """Main entry point."""
    print("Neural DSL - Test Coverage Report Generator")
    print("=" * 80)
    print()
    
    # Run pytest
    result = run_pytest_with_coverage()
    
    print()
    print("=" * 80)
    print("Test execution completed")
    print(f"Exit code: {result.returncode}")
    print()
    
    # Parse results
    stats = parse_pytest_output(result.stdout, result.stderr)
    coverage_data = parse_coverage_json()
    
    # Generate summary
    summary_md = generate_summary_markdown(stats, coverage_data, result.stdout, result.stderr)
    
    # Write to file
    output_file = Path("TEST_COVERAGE_SUMMARY.md")
    output_file.write_text(summary_md)
    
    print(f"✅ Summary written to: {output_file}")
    print()
    print("Summary:")
    print(f"  Total Tests: {stats['total']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    
    if coverage_data:
        totals = coverage_data.get("totals", {})
        num_statements = totals.get("num_statements", 0)
        covered_lines = totals.get("covered_lines", 0)
        if num_statements > 0:
            coverage_pct = (covered_lines / num_statements) * 100
            print(f"  Coverage: {coverage_pct:.1f}%")
    
    print()
    print("View detailed coverage report:")
    print("  HTML: htmlcov/index.html")
    print("  JSON: coverage.json")
    print()
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
