# Repository cleanup script to remove redundant documentation and artifacts
# This script removes 200+ redundant files including implementation summaries,
# quick reference docs, duplicate workflows, test artifacts, and cache directories

param(
    [switch]$DryRun = $false
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TotalRemoved = 0

function Remove-Items {
    param(
        [string[]]$Patterns,
        [string]$Description
    )
    
    $count = 0
    Write-Host "`n$Description" -ForegroundColor Cyan
    
    foreach ($pattern in $Patterns) {
        $path = Join-Path $RepoRoot $pattern
        if (Test-Path $path) {
            $relPath = $pattern
            if ($DryRun) {
                Write-Host "  [DRY RUN] Would remove: $relPath" -ForegroundColor Yellow
            } else {
                Write-Host "  - Removing: $relPath" -ForegroundColor Green
                Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
            }
            $count++
        }
    }
    
    return $count
}

function Remove-PatternFiles {
    param(
        [hashtable[]]$Patterns,
        [string]$Description
    )
    
    $count = 0
    Write-Host "`n$Description" -ForegroundColor Cyan
    
    foreach ($item in $Patterns) {
        $basePath = Join-Path $RepoRoot $item.Base
        if (Test-Path $basePath) {
            $files = Get-ChildItem -Path $basePath -Filter $item.Pattern -Recurse -File -ErrorAction SilentlyContinue
            foreach ($file in $files) {
                $relPath = $file.FullName.Replace($RepoRoot + "\", "")
                if ($DryRun) {
                    Write-Host "  [DRY RUN] Would remove: $relPath" -ForegroundColor Yellow
                } else {
                    Write-Host "  - Removing: $relPath" -ForegroundColor Green
                    Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
                }
                $count++
            }
        }
    }
    
    return $count
}

Write-Host "=" * 80 -ForegroundColor Magenta
Write-Host "NEURAL DSL REPOSITORY CLEANUP" -ForegroundColor Magenta
Write-Host "=" * 80 -ForegroundColor Magenta

if ($DryRun) {
    Write-Host "`n*** DRY RUN MODE - No files will be deleted ***`n" -ForegroundColor Yellow
}

# 1. Remove implementation summary documents (50+ files)
$implementationDocs = @(
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
    "tests\integration_tests\TEST_SUMMARY.md",
    "tests\benchmarks\SUMMARY.md",
    "tests\benchmarks\IMPLEMENTATION.md",
    "tests\TEST_COVERAGE_SUMMARY.md",
    "scripts\automation\IMPLEMENTATION_SUMMARY.md",
    "neural\visualization\IMPLEMENTATION_SUMMARY.md",
    "neural\tracking\IMPLEMENTATION_SUMMARY.md",
    "neural\profiling\IMPLEMENTATION_SUMMARY.md",
    "neural\parser\REFACTORING_SUMMARY.md",
    "neural\monitoring\IMPLEMENTATION_SUMMARY.md",
    "neural\federated\IMPLEMENTATION_SUMMARY.md",
    "neural\benchmarks\IMPLEMENTATION_COMPLETE.md",
    "neural\aquarium\src\components\terminal\IMPLEMENTATION_SUMMARY.md",
    "neural\aquarium\src\components\editor\SUMMARY.md",
    "neural\aquarium\src\components\editor\IMPLEMENTATION.md",
    "neural\aquarium\src\components\debugger\IMPLEMENTATION_SUMMARY.md",
    "neural\aquarium\electron\IMPLEMENTATION_COMPLETE.md",
    "neural\aquarium\PROJECT_SUMMARY.md",
    "neural\aquarium\PLUGIN_IMPLEMENTATION_SUMMARY.md",
    "neural\aquarium\PACKAGING_SUMMARY.md",
    "neural\aquarium\IMPLEMENTATION_SUMMARY.md",
    "neural\aquarium\IMPLEMENTATION_CHECKLIST.md",
    "neural\aquarium\IMPLEMENTATION.md",
    "neural\aquarium\HPO_IMPLEMENTATION.md",
    "neural\aquarium\EXPORT_IMPLEMENTATION_SUMMARY.md",
    "neural\aquarium\WELCOME_SCREEN_IMPLEMENTATION.md",
    "neural\api\IMPLEMENTATION_SUMMARY.md",
    "neural\ai\IMPLEMENTATION_SUMMARY.md",
    "examples\IMPLEMENTATION_SUMMARY.md",
    "docs\TRANSFORMER_DOCS_SUMMARY.md"
)
$TotalRemoved += Remove-Items -Patterns $implementationDocs -Description "1. Removing implementation summary documents"

# 2. Remove quick reference duplicates (30+ files)
$quickRefDocs = @(
    "TRANSFORMER_QUICK_REFERENCE.md",
    "POST_RELEASE_AUTOMATION_QUICK_REF.md",
    "DISTRIBUTION_QUICK_REF.md",
    "DEPENDENCY_QUICK_REF.md",
    "QUICK_START_AUTOMATION.md",
    "tests\benchmarks\QUICK_REFERENCE.md",
    "neural\tracking\QUICK_REFERENCE.md",
    "neural\integrations\QUICK_REFERENCE.md",
    "neural\cost\QUICK_REFERENCE.md",
    "neural\aquarium\QUICK_REFERENCE.md",
    "examples\EXAMPLES_QUICK_REF.md",
    "docs\mlops\QUICK_REFERENCE.md",
    "docs\aquarium\QUICK_REFERENCE.md",
    "docs\MARKETING_AUTOMATION_QUICK_REF.md",
    ".github\SECURITY_QUICK_REF.md"
)
$TotalRemoved += Remove-Items -Patterns $quickRefDocs -Description "2. Removing quick reference duplicates"

# 3. Remove redundant status/journal/plan documents
$statusDocs = @(
    "SETUP_STATUS.md",
    "CHANGES_SUMMARY.md",
    "BUG_FIXES.md",
    "CLEANUP_PLAN.md",
    "DISTRIBUTION_JOURNAL.md",
    "DISTRIBUTION_PLAN.md",
    "EXTRACTED_PROJECTS.md",
    "IMPORT_REFACTOR.md"
)
$TotalRemoved += Remove-Items -Patterns $statusDocs -Description "3. Removing redundant status/journal documents"

# 4. Remove duplicate release documents
$releaseDocs = @(
    "GITHUB_RELEASE_v0.3.0.md",
    "RELEASE_NOTES_v0.3.0.md",
    "RELEASE_VERIFICATION_v0.3.0.md",
    "MIGRATION_v0.3.0.md"
)
$TotalRemoved += Remove-Items -Patterns $releaseDocs -Description "4. Removing duplicate release documents"

# 5. Remove redundant guide documents
$redundantGuides = @(
    "AUTOMATION_GUIDE.md",
    "DEPENDENCY_GUIDE.md",
    "DEPLOYMENT_FEATURES.md",
    "ERROR_MESSAGES_GUIDE.md",
    "GITHUB_PUBLISHING_GUIDE.md",
    "MIGRATION_GUIDE_DEPENDENCIES.md",
    "WEBSITE_README.md"
)
$TotalRemoved += Remove-Items -Patterns $redundantGuides -Description "5. Removing redundant guide documents"

# 6. Remove duplicate docs in .github
$githubDocs = @(
    ".github\MARKETING_AUTOMATION_CHECKLIST.md",
    ".github\MARKETING_AUTOMATION_SUMMARY.md",
    ".github\SECURITY_BADGES.md",
    ".github\SECURITY_CHECKLIST.md",
    ".github\SECURITY_IMPLEMENTATION_SUMMARY.md",
    ".github\WORKFLOW_CONSOLIDATION.md",
    ".github\workflows\README_POST_RELEASE.md"
)
$TotalRemoved += Remove-Items -Patterns $githubDocs -Description "6. Removing duplicate .github docs"

# 7. Remove redundant workflow files
$workflowFiles = @(
    ".github\workflows\automated_release.yml",
    ".github\workflows\post_release.yml",
    ".github\workflows\periodic_tasks.yml",
    ".github\workflows\pytest-to-issues.yml",
    ".github\workflows\close-fixed-issues.yml"
)
$TotalRemoved += Remove-Items -Patterns $workflowFiles -Description "7. Removing redundant workflow files"

# 8. Remove duplicate install scripts
$installScripts = @(
    "_install_dev.py",
    "_setup_repo.py",
    "install_deps.py",
    "install.bat",
    "install_dev.bat"
)
$TotalRemoved += Remove-Items -Patterns $installScripts -Description "8. Removing duplicate install scripts"

# 9. Remove temporary/test scripts in root
$tempScripts = @(
    "repro_parser.py",
    "reproduce_issue.py"
)
$TotalRemoved += Remove-Items -Patterns $tempScripts -Description "9. Removing temporary test scripts"

# 10. Remove redundant requirements files
$reqFiles = @(
    "requirements-minimal.txt",
    "requirements-backends.txt",
    "requirements-viz.txt",
    "requirements-api.txt"
)
$TotalRemoved += Remove-Items -Patterns $reqFiles -Description "10. Removing redundant requirements files"

# 11. Remove Python cache directories
$cacheDirs = @(
    "__pycache__",
    "neural\aquarium\__pycache__",
    "neural\aquarium\src\components\settings\__pycache__",
    "neural\aquarium\src\plugins\__pycache__"
)
$TotalRemoved += Remove-Items -Patterns $cacheDirs -Description "11. Removing Python cache directories"

# 12. Remove .pyc files
$pycPatterns = @(
    @{Base = "."; Pattern = "*.pyc"},
    @{Base = "."; Pattern = "*.pyo"}
)
$TotalRemoved += Remove-PatternFiles -Patterns $pycPatterns -Description "12. Removing Python bytecode files"

# 13. Remove egg-info directory
$eggInfoDirs = @(
    "neural\neural_dsl.egg-info"
)
$TotalRemoved += Remove-Items -Patterns $eggInfoDirs -Description "13. Removing egg-info directories"

# 14. Remove duplicate scripts
$duplicateScripts = @(
    "scripts\close_fixed_issues.py",
    "scripts\complexity.py",
    "scripts\create_issues.py",
    "scripts\extract_projects.ps1",
    "scripts\extract_simple.ps1",
    "scripts\fix_cloud_execution_imports.py",
    "scripts\fix_cloud_tests.py",
    "scripts\fix_cloud_tests_round2.py",
    "scripts\fix_dashboard_resource_test.py",
    "scripts\fix_dashboard_test_logic.py",
    "scripts\fix_dashboard_tests.py",
    "scripts\fix_dashboard_tests_final.py",
    "scripts\fix_device_imports.py",
    "scripts\fix_device_tests.py",
    "scripts\fix_interactive_shell_paths.py",
    "scripts\fix_pretrained_tests.py",
    "scripts\force_add_skip.py",
    "scripts\github_publish_simple.ps1",
    "scripts\publish_to_github.ps1",
    "scripts\remove_trace_data_patch.py",
    "scripts\skip_failing_tests.py",
    "scripts\skip_hf_test.py"
)
$TotalRemoved += Remove-Items -Patterns $duplicateScripts -Description "14. Removing temporary fix scripts"

# 15. Remove redundant documentation in docs/
$docsDuplicates = @(
    "docs\BUILD_DOCS.md",
    "docs\DOCSTRING_GUIDE.md",
    "docs\MARKETING_AUTOMATION_DIAGRAM.md",
    "docs\MARKETING_AUTOMATION_GUIDE.md",
    "docs\MARKETING_AUTOMATION_SETUP.md",
    "docs\POST_RELEASE_AUTOMATION.md",
    "docs\RELEASE_QUICK_START.md",
    "docs\RELEASE_WORKFLOW.md",
    "docs\SECURITY_SETUP.md",
    "docs\DEPLOYMENT_QUICK_START.md"
)
$TotalRemoved += Remove-Items -Patterns $docsDuplicates -Description "15. Removing redundant documentation files"

# 16. Remove duplicate virtual environments
$venvDirs = @(
    ".venv312"
)
$TotalRemoved += Remove-Items -Patterns $venvDirs -Description "16. Removing duplicate virtual environments"

Write-Host "`n$('=' * 80)" -ForegroundColor Magenta
if ($DryRun) {
    Write-Host "DRY RUN COMPLETE: Would remove $TotalRemoved files/directories" -ForegroundColor Yellow
} else {
    Write-Host "CLEANUP COMPLETE: Removed $TotalRemoved files/directories" -ForegroundColor Green
}
Write-Host "$('=' * 80)" -ForegroundColor Magenta

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review changes with: git status"
Write-Host "2. Verify the repository still functions correctly"
Write-Host "3. Commit changes with: git add -A; git commit -m 'chore: clean up redundant documentation and artifacts'"

if ($DryRun) {
    Write-Host "`nRun without -DryRun flag to actually delete files" -ForegroundColor Yellow
}
