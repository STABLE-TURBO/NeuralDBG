"""
Test Post-Release Automation Setup

Validates that all required components for post-release automation are configured.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} missing: {filepath}")
        return False


def check_github_secret(secret_name: str, required: bool = False) -> bool:
    """Check if a GitHub secret should be configured."""
    status = "Required" if required else "Optional"
    print(f"  [{status}] {secret_name}")
    return True


def test_workflow_files():
    """Test that workflow files exist."""
    print("\n=== Workflow Files ===")
    checks = [
        check_file_exists(
            ".github/workflows/post_release_automation.yml",
            "Main workflow file"
        ),
        check_file_exists(
            ".github/workflows/README_POST_RELEASE.md",
            "Workflow README"
        ),
    ]
    return all(checks)


def test_script_files():
    """Test that helper scripts exist."""
    print("\n=== Helper Scripts ===")
    checks = [
        check_file_exists(
            "scripts/automation/post_release_helper.py",
            "Post-release helper script"
        ),
    ]
    return all(checks)


def test_documentation():
    """Test that documentation files exist."""
    print("\n=== Documentation ===")
    checks = [
        check_file_exists(
            "docs/POST_RELEASE_AUTOMATION.md",
            "Full documentation"
        ),
        check_file_exists(
            "POST_RELEASE_AUTOMATION_QUICK_REF.md",
            "Quick reference guide"
        ),
    ]
    return all(checks)


def test_version_files():
    """Test that version files exist and are readable."""
    print("\n=== Version Files ===")
    checks = []
    
    # Check setup.py
    if check_file_exists("setup.py", "Setup file"):
        try:
            with open("setup.py", "r") as f:
                content = f.read()
                if 'version=' in content:
                    print("  ✓ Version string found in setup.py")
                    checks.append(True)
                else:
                    print("  ✗ Version string not found in setup.py")
                    checks.append(False)
        except Exception as e:
            print(f"  ✗ Error reading setup.py: {e}")
            checks.append(False)
    else:
        checks.append(False)
    
    # Check neural/__init__.py
    if check_file_exists("neural/__init__.py", "Package init file"):
        try:
            with open("neural/__init__.py", "r") as f:
                content = f.read()
                if '__version__' in content:
                    print("  ✓ __version__ found in neural/__init__.py")
                    checks.append(True)
                else:
                    print("  ✗ __version__ not found in neural/__init__.py")
                    checks.append(False)
        except Exception as e:
            print(f"  ✗ Error reading neural/__init__.py: {e}")
            checks.append(False)
    else:
        checks.append(False)
    
    return all(checks)


def test_changelog():
    """Test that CHANGELOG.md exists and has proper format."""
    print("\n=== Changelog ===")
    if not check_file_exists("CHANGELOG.md", "Changelog file"):
        return False
    
    try:
        with open("CHANGELOG.md", "r") as f:
            content = f.read()
            if "## [" in content:
                print("  ✓ Version headers found in CHANGELOG.md")
                return True
            else:
                print("  ✗ No version headers found in CHANGELOG.md")
                print("     Expected format: ## [X.Y.Z]")
                return False
    except Exception as e:
        print(f"  ✗ Error reading CHANGELOG.md: {e}")
        return False


def test_github_secrets():
    """Display required and optional GitHub secrets."""
    print("\n=== GitHub Secrets ===")
    print("\nRequired:")
    check_github_secret("GITHUB_TOKEN", required=True)
    print("  (Auto-provided by GitHub Actions)")
    
    print("\nOptional (for enhanced features):")
    check_github_secret("NETLIFY_BUILD_HOOK")
    print("    Get from: Netlify Dashboard → Build Hooks")
    check_github_secret("VERCEL_DEPLOY_HOOK")
    print("    Get from: Vercel Dashboard → Deploy Hooks")
    check_github_secret("DISCORD_WEBHOOK_URL")
    print("    Get from: Discord → Channel Settings → Webhooks")
    
    print("\n  To configure: Repository → Settings → Secrets → Actions")
    return True


def test_permissions():
    """Display required permissions."""
    print("\n=== GitHub Permissions ===")
    print("Required workflow permissions:")
    print("  ✓ contents: write (to commit version bump)")
    print("  ✓ discussions: write (to create announcements)")
    print("  ✓ issues: write (to create planning issues)")
    print("  ✓ pull-requests: write (for future use)")
    print("\nTo configure:")
    print("  Repository → Settings → Actions → General → Workflow permissions")
    print("  ☑ Read and write permissions")
    return True


def test_discussions_enabled():
    """Check if discussions feature recommendation."""
    print("\n=== GitHub Discussions ===")
    print("For discussion announcements to work, enable Discussions:")
    print("  Repository → Settings → Features → ☑ Discussions")
    return True


def test_dependencies():
    """Test that required Python dependencies are available."""
    print("\n=== Python Dependencies ===")
    checks = []
    
    dependencies = [
        ("requests", "For webhook calls and API requests"),
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
            checks.append(True)
        except ImportError:
            print(f"✗ {package} not installed: {description}")
            print(f"  Install with: pip install {package}")
            checks.append(False)
    
    return all(checks)


def main():
    """Run all tests."""
    print("=" * 70)
    print("Post-Release Automation Setup Verification")
    print("=" * 70)
    
    results = {
        "Workflow Files": test_workflow_files(),
        "Helper Scripts": test_script_files(),
        "Documentation": test_documentation(),
        "Version Files": test_version_files(),
        "Changelog": test_changelog(),
        "Python Dependencies": test_dependencies(),
    }
    
    # Informational checks (always pass)
    test_github_secrets()
    test_permissions()
    test_discussions_enabled()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All checks passed! Post-release automation is ready.")
        print("\nNext steps:")
        print("1. Configure optional GitHub secrets (if needed)")
        print("2. Enable Discussions in repository settings")
        print("3. Verify workflow permissions")
        print("4. Test with manual workflow dispatch")
        print("\nSee: POST_RELEASE_AUTOMATION_QUICK_REF.md for usage")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nSee: docs/POST_RELEASE_AUTOMATION.md for setup instructions")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
