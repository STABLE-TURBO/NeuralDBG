#!/bin/bash
# Repository cleanup script to remove redundant documentation and artifacts

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOTAL_REMOVED=0
DRY_RUN=false

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "*** DRY RUN MODE - No files will be deleted ***"
    echo ""
fi

remove_files() {
    local description="$1"
    shift
    local files=("$@")
    local count=0
    
    echo ""
    echo "$description"
    
    for file in "${files[@]}"; do
        local path="$REPO_ROOT/$file"
        if [[ -e "$path" ]]; then
            if $DRY_RUN; then
                echo "  [DRY RUN] Would remove: $file"
            else
                echo "  - Removing: $file"
                rm -rf "$path"
            fi
            ((count++))
        fi
    done
    
    echo "$count"
}

echo "================================================================================"
echo "NEURAL DSL REPOSITORY CLEANUP"
echo "================================================================================"

# 1. Remove implementation summary documents
count=$(remove_files "1. Removing implementation summary documents" \
    "AQUARIUM_IMPLEMENTATION_SUMMARY.md" \
    "BENCHMARKS_IMPLEMENTATION_SUMMARY.md" \
    "CLOUD_IMPROVEMENTS_SUMMARY.md" \
    "COST_OPTIMIZATION_IMPLEMENTATION.md" \
    "DATA_VERSIONING_IMPLEMENTATION.md" \
    "DEPENDENCY_OPTIMIZATION_SUMMARY.md" \
    "DOCUMENTATION_SUMMARY.md" \
    "IMPLEMENTATION_CHECKLIST.md" \
    "IMPLEMENTATION_COMPLETE.md" \
    "IMPLEMENTATION_SUMMARY.md" \
    "INTEGRATIONS_SUMMARY.md" \
    "INTEGRATION_IMPLEMENTATION.md" \
    "MARKETPLACE_IMPLEMENTATION.md" \
    "MARKETPLACE_SUMMARY.md" \
    "MLOPS_IMPLEMENTATION.md" \
    "MULTIHEADATTENTION_IMPLEMENTATION.md" \
    "NEURAL_API_IMPLEMENTATION.md" \
    "PERFORMANCE_IMPLEMENTATION.md" \
    "POSITIONAL_ENCODING_IMPLEMENTATION.md" \
    "POST_RELEASE_IMPLEMENTATION_SUMMARY.md" \
    "TEAMS_IMPLEMENTATION.md" \
    "TRANSFORMER_DECODER_IMPLEMENTATION.md" \
    "TRANSFORMER_ENHANCEMENTS.md" \
    "V0.3.0_RELEASE_SUMMARY.md" \
    "WEBSITE_IMPLEMENTATION_SUMMARY.md" \
    "tests/integration_tests/TEST_SUMMARY.md" \
    "tests/benchmarks/SUMMARY.md" \
    "tests/benchmarks/IMPLEMENTATION.md" \
    "tests/TEST_COVERAGE_SUMMARY.md" \
    "scripts/automation/IMPLEMENTATION_SUMMARY.md" \
    "neural/visualization/IMPLEMENTATION_SUMMARY.md" \
    "neural/tracking/IMPLEMENTATION_SUMMARY.md" \
    "neural/profiling/IMPLEMENTATION_SUMMARY.md" \
    "neural/parser/REFACTORING_SUMMARY.md" \
    "neural/monitoring/IMPLEMENTATION_SUMMARY.md" \
    "neural/federated/IMPLEMENTATION_SUMMARY.md" \
    "neural/benchmarks/IMPLEMENTATION_COMPLETE.md" \
    "neural/aquarium/src/components/terminal/IMPLEMENTATION_SUMMARY.md" \
    "neural/aquarium/src/components/editor/SUMMARY.md" \
    "neural/aquarium/src/components/editor/IMPLEMENTATION.md" \
    "neural/aquarium/src/components/debugger/IMPLEMENTATION_SUMMARY.md" \
    "neural/aquarium/electron/IMPLEMENTATION_COMPLETE.md" \
    "neural/aquarium/PROJECT_SUMMARY.md" \
    "neural/aquarium/PLUGIN_IMPLEMENTATION_SUMMARY.md" \
    "neural/aquarium/PACKAGING_SUMMARY.md" \
    "neural/aquarium/IMPLEMENTATION_SUMMARY.md" \
    "neural/aquarium/IMPLEMENTATION_CHECKLIST.md" \
    "neural/aquarium/IMPLEMENTATION.md" \
    "neural/aquarium/HPO_IMPLEMENTATION.md" \
    "neural/aquarium/EXPORT_IMPLEMENTATION_SUMMARY.md" \
    "neural/aquarium/WELCOME_SCREEN_IMPLEMENTATION.md" \
    "neural/api/IMPLEMENTATION_SUMMARY.md" \
    "neural/ai/IMPLEMENTATION_SUMMARY.md" \
    "examples/IMPLEMENTATION_SUMMARY.md" \
    "docs/TRANSFORMER_DOCS_SUMMARY.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 2. Remove quick reference duplicates
count=$(remove_files "2. Removing quick reference duplicates" \
    "TRANSFORMER_QUICK_REFERENCE.md" \
    "POST_RELEASE_AUTOMATION_QUICK_REF.md" \
    "DISTRIBUTION_QUICK_REF.md" \
    "DEPENDENCY_QUICK_REF.md" \
    "QUICK_START_AUTOMATION.md" \
    "tests/benchmarks/QUICK_REFERENCE.md" \
    "neural/tracking/QUICK_REFERENCE.md" \
    "neural/integrations/QUICK_REFERENCE.md" \
    "neural/cost/QUICK_REFERENCE.md" \
    "neural/aquarium/QUICK_REFERENCE.md" \
    "examples/EXAMPLES_QUICK_REF.md" \
    "docs/mlops/QUICK_REFERENCE.md" \
    "docs/aquarium/QUICK_REFERENCE.md" \
    "docs/MARKETING_AUTOMATION_QUICK_REF.md" \
    ".github/SECURITY_QUICK_REF.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 3. Remove redundant status/journal documents
count=$(remove_files "3. Removing redundant status/journal documents" \
    "SETUP_STATUS.md" \
    "CHANGES_SUMMARY.md" \
    "BUG_FIXES.md" \
    "CLEANUP_PLAN.md" \
    "DISTRIBUTION_JOURNAL.md" \
    "DISTRIBUTION_PLAN.md" \
    "EXTRACTED_PROJECTS.md" \
    "IMPORT_REFACTOR.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 4. Remove duplicate release documents
count=$(remove_files "4. Removing duplicate release documents" \
    "GITHUB_RELEASE_v0.3.0.md" \
    "RELEASE_NOTES_v0.3.0.md" \
    "RELEASE_VERIFICATION_v0.3.0.md" \
    "MIGRATION_v0.3.0.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 5. Remove redundant guide documents
count=$(remove_files "5. Removing redundant guide documents" \
    "AUTOMATION_GUIDE.md" \
    "DEPENDENCY_GUIDE.md" \
    "DEPLOYMENT_FEATURES.md" \
    "ERROR_MESSAGES_GUIDE.md" \
    "GITHUB_PUBLISHING_GUIDE.md" \
    "MIGRATION_GUIDE_DEPENDENCIES.md" \
    "WEBSITE_README.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 6. Remove duplicate docs in .github
count=$(remove_files "6. Removing duplicate .github docs" \
    ".github/MARKETING_AUTOMATION_CHECKLIST.md" \
    ".github/MARKETING_AUTOMATION_SUMMARY.md" \
    ".github/SECURITY_BADGES.md" \
    ".github/SECURITY_CHECKLIST.md" \
    ".github/SECURITY_IMPLEMENTATION_SUMMARY.md" \
    ".github/WORKFLOW_CONSOLIDATION.md" \
    ".github/workflows/README_POST_RELEASE.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 7. Remove redundant workflow files
count=$(remove_files "7. Removing redundant workflow files" \
    ".github/workflows/automated_release.yml" \
    ".github/workflows/post_release.yml" \
    ".github/workflows/periodic_tasks.yml" \
    ".github/workflows/pytest-to-issues.yml" \
    ".github/workflows/close-fixed-issues.yml")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 8. Remove duplicate install scripts
count=$(remove_files "8. Removing duplicate install scripts" \
    "_install_dev.py" \
    "_setup_repo.py" \
    "install_deps.py" \
    "install.bat" \
    "install_dev.bat")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 9. Remove temporary test scripts
count=$(remove_files "9. Removing temporary test scripts" \
    "repro_parser.py" \
    "reproduce_issue.py")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 10. Remove redundant requirements files
count=$(remove_files "10. Removing redundant requirements files" \
    "requirements-minimal.txt" \
    "requirements-backends.txt" \
    "requirements-viz.txt" \
    "requirements-api.txt")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 11. Remove Python cache directories
count=$(remove_files "11. Removing Python cache directories" \
    "__pycache__" \
    "neural/aquarium/__pycache__" \
    "neural/aquarium/src/components/settings/__pycache__" \
    "neural/aquarium/src/plugins/__pycache__")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 12. Remove .pyc files
echo ""
echo "12. Removing Python bytecode files"
if $DRY_RUN; then
    pyc_count=$(find "$REPO_ROOT" -name "*.pyc" -o -name "*.pyo" | wc -l)
    echo "  [DRY RUN] Would remove $pyc_count .pyc/.pyo files"
else
    pyc_count=$(find "$REPO_ROOT" -name "*.pyc" -o -name "*.pyo" -type f -delete -print | wc -l)
    echo "  - Removed $pyc_count .pyc/.pyo files"
fi
TOTAL_REMOVED=$((TOTAL_REMOVED + pyc_count))

# 13. Remove egg-info directory
count=$(remove_files "13. Removing egg-info directories" \
    "neural/neural_dsl.egg-info")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 14. Remove duplicate scripts
count=$(remove_files "14. Removing temporary fix scripts" \
    "scripts/close_fixed_issues.py" \
    "scripts/complexity.py" \
    "scripts/create_issues.py" \
    "scripts/extract_projects.ps1" \
    "scripts/extract_simple.ps1" \
    "scripts/fix_cloud_execution_imports.py" \
    "scripts/fix_cloud_tests.py" \
    "scripts/fix_cloud_tests_round2.py" \
    "scripts/fix_dashboard_resource_test.py" \
    "scripts/fix_dashboard_test_logic.py" \
    "scripts/fix_dashboard_tests.py" \
    "scripts/fix_dashboard_tests_final.py" \
    "scripts/fix_device_imports.py" \
    "scripts/fix_device_tests.py" \
    "scripts/fix_interactive_shell_paths.py" \
    "scripts/fix_pretrained_tests.py" \
    "scripts/force_add_skip.py" \
    "scripts/github_publish_simple.ps1" \
    "scripts/publish_to_github.ps1" \
    "scripts/remove_trace_data_patch.py" \
    "scripts/skip_failing_tests.py" \
    "scripts/skip_hf_test.py")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 15. Remove redundant documentation
count=$(remove_files "15. Removing redundant documentation files" \
    "docs/BUILD_DOCS.md" \
    "docs/DOCSTRING_GUIDE.md" \
    "docs/MARKETING_AUTOMATION_DIAGRAM.md" \
    "docs/MARKETING_AUTOMATION_GUIDE.md" \
    "docs/MARKETING_AUTOMATION_SETUP.md" \
    "docs/POST_RELEASE_AUTOMATION.md" \
    "docs/RELEASE_QUICK_START.md" \
    "docs/RELEASE_WORKFLOW.md" \
    "docs/SECURITY_SETUP.md" \
    "docs/DEPLOYMENT_QUICK_START.md")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

# 16. Remove duplicate virtual environments
count=$(remove_files "16. Removing duplicate virtual environments" \
    ".venv312")
TOTAL_REMOVED=$((TOTAL_REMOVED + count))

echo ""
echo "================================================================================"
if $DRY_RUN; then
    echo "DRY RUN COMPLETE: Would remove $TOTAL_REMOVED files/directories"
else
    echo "CLEANUP COMPLETE: Removed $TOTAL_REMOVED files/directories"
fi
echo "================================================================================"

echo ""
echo "Next steps:"
echo "1. Review changes with: git status"
echo "2. Verify the repository still functions correctly"
echo "3. Commit changes with: git add -A && git commit -m 'chore: clean up redundant documentation and artifacts'"

if $DRY_RUN; then
    echo ""
    echo "Run without --dry-run flag to actually delete files"
fi
