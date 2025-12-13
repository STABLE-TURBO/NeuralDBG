#!/usr/bin/env python3
"""
Verification Script for Neural DSL Setup

Checks that all automation and AI components are properly set up.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"⚠️  {description}: {module_name} - {e}")
        return False

def main():
    """Run verification checks."""
    print("=" * 70)
    print("Neural DSL Setup Verification")
    print("=" * 70)
    print()
    
    all_checks = []
    
    # Check AI Integration Files
    print("🤖 AI Integration Files:")
    print("-" * 70)
    all_checks.append(check_file_exists("neural/ai/natural_language_processor.py", "Natural Language Processor"))
    all_checks.append(check_file_exists("neural/ai/llm_integration.py", "LLM Integration"))
    all_checks.append(check_file_exists("neural/ai/multi_language.py", "Multi-Language Support"))
    all_checks.append(check_file_exists("neural/ai/ai_assistant.py", "AI Assistant"))
    all_checks.append(check_file_exists("neural/ai/README.md", "AI README"))
    print()
    
    # Check Automation Scripts
    print("🔄 Automation Scripts:")
    print("-" * 70)
    all_checks.append(check_file_exists("scripts/automation/release_automation.py", "Release Automation"))
    all_checks.append(check_file_exists("scripts/automation/example_validator.py", "Example Validator"))
    all_checks.append(check_file_exists("scripts/automation/test_automation.py", "Test Automation"))
    all_checks.append(check_file_exists("scripts/automation/master_automation.py", "Master Automation"))
    print()
    
    # Check GitHub Workflows
    print("⚙️  GitHub Actions:")
    print("-" * 70)
    all_checks.append(check_file_exists(".github/workflows/automated_release.yml", "Automated Release Workflow"))
    all_checks.append(check_file_exists(".github/workflows/periodic_tasks.yml", "Periodic Tasks Workflow"))
    print()
    
    # Check Documentation
    print("📚 Documentation:")
    print("-" * 70)
    all_checks.append(check_file_exists("AUTOMATION_GUIDE.md", "Automation Guide"))
    all_checks.append(check_file_exists("CONTRIBUTING.md", "Contributing Guide"))
    all_checks.append(check_file_exists("WHATS_NEW.md", "What's New"))
    all_checks.append(check_file_exists("docs/ai_integration_guide.md", "AI Integration Guide"))
    print()
    
    # Check Imports
    print("🐍 Python Imports:")
    print("-" * 70)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Try importing AI modules (may fail if dependencies missing, that's OK)
    try:
        from neural.ai.natural_language_processor import NaturalLanguageProcessor
        print("✅ Natural Language Processor: Can import")
        all_checks.append(True)
    except Exception as e:
        print(f"⚠️  Natural Language Processor: {e}")
        all_checks.append(False)
    
    try:
        from neural.ai.ai_assistant import NeuralAIAssistant
        print("✅ AI Assistant: Can import")
        all_checks.append(True)
    except Exception as e:
        print(f"⚠️  AI Assistant: {e}")
        all_checks.append(False)
    
    print()
    
    # Summary
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Summary: {passed}/{total} checks passed ({percentage:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("✅ All checks passed! Setup is complete.")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  Most checks passed. Some optional components may be missing.")
        return 0
    else:
        print("❌ Some critical components are missing. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

