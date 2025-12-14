"""Repository cleanup script to remove redundant documentation and artifacts."""

import os
import shutil
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).parent.parent


def remove_files(file_patterns: List[str], description: str) -> int:
    """Remove files matching patterns and return count."""
    count = 0
    print(f"\n{description}:")
    
    for pattern in file_patterns:
        path = REPO_ROOT / pattern
        if path.exists():
            if path.is_file():
                print(f"  - Removing: {pattern}")
                path.unlink()
                count += 1
            elif path.is_dir():
                print(f"  - Removing directory: {pattern}")
                shutil.rmtree(path)
                count += 1
    
    return count


def remove_pattern_files(pattern_dirs: List[tuple], description: str) -> int:
    """Remove files matching glob patterns."""
    count = 0
    print(f"\n{description}:")
    
    for base_dir, pattern in pattern_dirs:
        base_path = REPO_ROOT / base_dir
        if base_path.exists():
            for file_path in base_path.rglob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(REPO_ROOT)
                    print(f"  - Removing: {rel_path}")
                    file_path.unlink()
                    count += 1
    
    return count


def main():
    """Execute cleanup operations."""
    print("=" * 80)
    print("NEURAL DSL REPOSITORY CLEANUP")
    print("=" * 80)
    
    total_removed = 0
    
    # 1. Remove implementation summary documents (50+ files)
    implementation_docs = [
        "AQUARIUM_IMPLEMENTATION_SUMMARY.md",
        "BENCHMARKS_IMPLEMENTATION_SUMMARY.md",
        "CLOUD_IMPROVEMENTS_SUMMARY.md",
        "COST_OPTIMIZATION_IMPLEMENTATION.md",
        "DATA_VERSIONING_IMPLEMENTATION.md",
        "DEPENDENCY_OPTIMIZATION_SUMMARY.md",
        "DOCUMENTATION_SUMMARY.md",
        "IMPLEMENTATION_CHECKLIST.md",
        "IMPLEMENTATION_COMPLETE.md",
        "IMPLEMENTATION_SUMMARY.md",
        "INTEGRATIONS_SUMMARY.md",
        "INTEGRATION_IMPLEMENTATION.md",
        "MARKETPLACE_IMPLEMENTATION.md",
        "MARKETPLACE_SUMMARY.md",
        "MLOPS_IMPLEMENTATION.md",
        "MULTIHEADATTENTION_IMPLEMENTATION.md",
        "NEURAL_API_IMPLEMENTATION.md",
        "PERFORMANCE_IMPLEMENTATION.md",
        "POSITIONAL_ENCODING_IMPLEMENTATION.md",
        "POST_RELEASE_IMPLEMENTATION_SUMMARY.md",
        "TEAMS_IMPLEMENTATION.md",
        "TRANSFORMER_DECODER_IMPLEMENTATION.md",
        "TRANSFORMER_ENHANCEMENTS.md",
        "V0.3.0_RELEASE_SUMMARY.md",
        "WEBSITE_IMPLEMENTATION_SUMMARY.md",
        "tests/integration_tests/TEST_SUMMARY.md",
        "tests/benchmarks/SUMMARY.md",
        "tests/benchmarks/IMPLEMENTATION.md",
        "tests/TEST_COVERAGE_SUMMARY.md",
        "scripts/automation/IMPLEMENTATION_SUMMARY.md",
        "neural/visualization/IMPLEMENTATION_SUMMARY.md",
        "neural/tracking/IMPLEMENTATION_SUMMARY.md",
        "neural/profiling/IMPLEMENTATION_SUMMARY.md",
        "neural/parser/REFACTORING_SUMMARY.md",
        "neural/monitoring/IMPLEMENTATION_SUMMARY.md",
        "neural/federated/IMPLEMENTATION_SUMMARY.md",
        "neural/benchmarks/IMPLEMENTATION_COMPLETE.md",
        "neural/aquarium/src/components/terminal/IMPLEMENTATION_SUMMARY.md",
        "neural/aquarium/src/components/editor/SUMMARY.md",
        "neural/aquarium/src/components/editor/IMPLEMENTATION.md",
        "neural/aquarium/src/components/debugger/IMPLEMENTATION_SUMMARY.md",
        "neural/aquarium/electron/IMPLEMENTATION_COMPLETE.md",
        "neural/aquarium/PROJECT_SUMMARY.md",
        "neural/aquarium/PLUGIN_IMPLEMENTATION_SUMMARY.md",
        "neural/aquarium/PACKAGING_SUMMARY.md",
        "neural/aquarium/IMPLEMENTATION_SUMMARY.md",
        "neural/aquarium/IMPLEMENTATION_CHECKLIST.md",
        "neural/aquarium/IMPLEMENTATION.md",
        "neural/aquarium/HPO_IMPLEMENTATION.md",
        "neural/aquarium/EXPORT_IMPLEMENTATION_SUMMARY.md",
        "neural/aquarium/WELCOME_SCREEN_IMPLEMENTATION.md",
        "neural/api/IMPLEMENTATION_SUMMARY.md",
        "neural/ai/IMPLEMENTATION_SUMMARY.md",
        "examples/IMPLEMENTATION_SUMMARY.md",
        "docs/TRANSFORMER_DOCS_SUMMARY.md",
    ]
    total_removed += remove_files(implementation_docs, "1. Removing implementation summary documents")
    
    # 2. Remove quick reference duplicates (30+ files)
    quick_ref_docs = [
        "TRANSFORMER_QUICK_REFERENCE.md",
        "POST_RELEASE_AUTOMATION_QUICK_REF.md",
        "DISTRIBUTION_QUICK_REF.md",
        "DEPENDENCY_QUICK_REF.md",
        "QUICK_START_AUTOMATION.md",
        "tests/benchmarks/QUICK_REFERENCE.md",
        "neural/tracking/QUICK_REFERENCE.md",
        "neural/integrations/QUICK_REFERENCE.md",
        "neural/cost/QUICK_REFERENCE.md",
        "neural/aquarium/QUICK_REFERENCE.md",
        "examples/EXAMPLES_QUICK_REF.md",
        "docs/mlops/QUICK_REFERENCE.md",
        "docs/aquarium/QUICK_REFERENCE.md",
        "docs/MARKETING_AUTOMATION_QUICK_REF.md",
        ".github/SECURITY_QUICK_REF.md",
    ]
    total_removed += remove_files(quick_ref_docs, "2. Removing quick reference duplicates")
    
    # 3. Remove redundant status/journal/plan documents
    status_docs = [
        "SETUP_STATUS.md",
        "CHANGES_SUMMARY.md",
        "BUG_FIXES.md",
        "CLEANUP_PLAN.md",
        "DISTRIBUTION_JOURNAL.md",
        "DISTRIBUTION_PLAN.md",
        "EXTRACTED_PROJECTS.md",
        "IMPORT_REFACTOR.md",
    ]
    total_removed += remove_files(status_docs, "3. Removing redundant status/journal documents")
    
    # 4. Remove duplicate release documents
    release_docs = [
        "GITHUB_RELEASE_v0.3.0.md",
        "RELEASE_NOTES_v0.3.0.md",
        "RELEASE_VERIFICATION_v0.3.0.md",
        "MIGRATION_v0.3.0.md",
    ]
    total_removed += remove_files(release_docs, "4. Removing duplicate release documents")
    
    # 5. Remove redundant guide documents (content should be in docs/)
    redundant_guides = [
        "AUTOMATION_GUIDE.md",
        "DEPENDENCY_GUIDE.md",
        "DEPLOYMENT_FEATURES.md",
        "ERROR_MESSAGES_GUIDE.md",
        "GITHUB_PUBLISHING_GUIDE.md",
        "MIGRATION_GUIDE_DEPENDENCIES.md",
        "WEBSITE_README.md",
    ]
    total_removed += remove_files(redundant_guides, "5. Removing redundant guide documents")
    
    # 6. Remove duplicate docs in .github
    github_docs = [
        ".github/MARKETING_AUTOMATION_CHECKLIST.md",
        ".github/MARKETING_AUTOMATION_SUMMARY.md",
        ".github/SECURITY_BADGES.md",
        ".github/SECURITY_CHECKLIST.md",
        ".github/SECURITY_IMPLEMENTATION_SUMMARY.md",
        ".github/WORKFLOW_CONSOLIDATION.md",
        ".github/workflows/README_POST_RELEASE.md",
    ]
    total_removed += remove_files(github_docs, "6. Removing duplicate .github docs")
    
    # 7. Remove redundant workflow files
    workflow_files = [
        ".github/workflows/automated_release.yml",
        ".github/workflows/post_release.yml",
        ".github/workflows/periodic_tasks.yml",
        ".github/workflows/pytest-to-issues.yml",
        ".github/workflows/close-fixed-issues.yml",
    ]
    total_removed += remove_files(workflow_files, "7. Removing redundant workflow files")
    
    # 8. Remove duplicate install scripts
    install_scripts = [
        "_install_dev.py",
        "_setup_repo.py",
        "install_deps.py",
        "install.bat",
        "install_dev.bat",
    ]
    total_removed += remove_files(install_scripts, "8. Removing duplicate install scripts")
    
    # 9. Remove temporary/test scripts in root
    temp_scripts = [
        "repro_parser.py",
        "reproduce_issue.py",
    ]
    total_removed += remove_files(temp_scripts, "9. Removing temporary test scripts")
    
    # 10. Remove redundant requirements files
    req_files = [
        "requirements-minimal.txt",
        "requirements-backends.txt",
        "requirements-viz.txt",
        "requirements-api.txt",
    ]
    total_removed += remove_files(req_files, "10. Removing redundant requirements files")
    
    # 11. Remove Python cache directories
    cache_dirs = [
        "__pycache__",
        "neural/aquarium/__pycache__",
        "neural/aquarium/src/components/settings/__pycache__",
        "neural/aquarium/src/plugins/__pycache__",
    ]
    total_removed += remove_files(cache_dirs, "11. Removing Python cache directories")
    
    # 12. Remove .pyc files throughout the repo
    total_removed += remove_pattern_files(
        [(".", "**/*.pyc"), (".", "**/*.pyo")],
        "12. Removing Python bytecode files"
    )
    
    # 13. Remove egg-info directory
    egg_info_dirs = [
        "neural/neural_dsl.egg-info",
    ]
    total_removed += remove_files(egg_info_dirs, "13. Removing egg-info directories")
    
    # 14. Remove duplicate scripts
    duplicate_scripts = [
        "scripts/close_fixed_issues.py",
        "scripts/complexity.py",
        "scripts/create_issues.py",
        "scripts/extract_projects.ps1",
        "scripts/extract_simple.ps1",
        "scripts/fix_cloud_execution_imports.py",
        "scripts/fix_cloud_tests.py",
        "scripts/fix_cloud_tests_round2.py",
        "scripts/fix_dashboard_resource_test.py",
        "scripts/fix_dashboard_test_logic.py",
        "scripts/fix_dashboard_tests.py",
        "scripts/fix_dashboard_tests_final.py",
        "scripts/fix_device_imports.py",
        "scripts/fix_device_tests.py",
        "scripts/fix_interactive_shell_paths.py",
        "scripts/fix_pretrained_tests.py",
        "scripts/force_add_skip.py",
        "scripts/github_publish_simple.ps1",
        "scripts/publish_to_github.ps1",
        "scripts/remove_trace_data_patch.py",
        "scripts/skip_failing_tests.py",
        "scripts/skip_hf_test.py",
    ]
    total_removed += remove_files(duplicate_scripts, "14. Removing temporary fix scripts")
    
    # 15. Remove redundant documentation in docs/
    docs_duplicates = [
        "docs/BUILD_DOCS.md",
        "docs/DOCSTRING_GUIDE.md",
        "docs/MARKETING_AUTOMATION_DIAGRAM.md",
        "docs/MARKETING_AUTOMATION_GUIDE.md",
        "docs/MARKETING_AUTOMATION_SETUP.md",
        "docs/POST_RELEASE_AUTOMATION.md",
        "docs/RELEASE_QUICK_START.md",
        "docs/RELEASE_WORKFLOW.md",
        "docs/SECURITY_SETUP.md",
        "docs/DEPLOYMENT_QUICK_START.md",
    ]
    total_removed += remove_files(docs_duplicates, "15. Removing redundant documentation files")
    
    # 16. Remove .venv312 if it exists (duplicate venv)
    venv_dirs = [
        ".venv312",
    ]
    total_removed += remove_files(venv_dirs, "16. Removing duplicate virtual environments")
    
    print("\n" + "=" * 80)
    print(f"CLEANUP COMPLETE: Removed {total_removed} files/directories")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review changes with: git status")
    print("2. Verify the repository still functions correctly")
    print("3. Commit changes with: git add -A && git commit -m 'chore: clean up redundant documentation and artifacts'")


if __name__ == "__main__":
    main()
