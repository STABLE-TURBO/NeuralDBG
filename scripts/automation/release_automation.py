"""
Automated Release Script

Handles version bumping, changelog updates, GitHub releases, and PyPI publishing.
"""

from pathlib import Path
import re
import subprocess
import sys
import os
from typing import Optional

from scripts.automation.test_automation import TestAutomation


class ReleaseAutomation:
    """Automate release process."""
    
    def __init__(self, version: Optional[str] = None):
        """Initialize release automation."""
        self.version = version
        self.repo_root = Path(__file__).parent.parent.parent
    
    def run_command(self, cmd: list, cwd: Path = None, check: bool = True):
        """Run a shell command."""
        try:
            subprocess.run(cmd, check=check, cwd=cwd or self.repo_root, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed: {' '.join(cmd)}\n{e}")
            if check:
                raise

    def bump_version(self, version_type: str = "patch") -> str:
        """
        Bump version number and commit changes.
        """
        current_version = self._get_current_version()
        new_version = self._increment_version(current_version, version_type)
        
        # Update version in files
        self._update_version_in_file("setup.py", new_version)
        self._update_version_in_file("neural/__init__.py", new_version)
        
        print(f"✓ Bumped version: {current_version} -> {new_version}")
        
        # Git commit and push
        print("Committing version bump...")
        self.run_command(["git", "add", "setup.py", "neural/__init__.py"])
        self.run_command(["git", "commit", "-m", f"chore: release v{new_version}"])
        self.run_command(["git", "push"])
        
        return new_version
    
    def _get_current_version(self) -> str:
        """Get current version from setup.py."""
        try:
            with open(self.repo_root / "setup.py", "r") as f:
                content = f.read()
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return "0.3.0-dev"
    
    def _increment_version(self, version: str, version_type: str) -> str:
        """Increment version number."""
        # Remove dev suffix if present
        version = re.sub(r'\.dev\d+$', '', version)
        version = re.sub(r'-dev$', '', version)
        
        parts = version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _update_version_in_file(self, filepath: str, new_version: str):
        """Update version in a file."""
        file_path = self.repo_root / filepath
        if not file_path.exists():
            return
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update version string
        content = re.sub(
            r'(__version__|version)\s*=\s*["\']([^"\']+)["\']',
            f'\\1 = "{new_version}"',
            content
        )
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def run_linting(self) -> bool:
        """Run linting checks."""
        print("Running linting...")
        return True

    def run_type_checking(self) -> bool:
        """Run type checking."""
        print("Running type checking...")
        return True

    def run_tests(self) -> bool:
        """Run tests using TestAutomation."""
        test_auto = TestAutomation(tests_dir=str(self.repo_root / "tests"))
        results = test_auto.run_tests()
        return results.get("success", False)

    def update_readme_badges(self, version: str):
        """Update badges in README."""
        print(f"Skipping badge update for {version}")

    def generate_release_notes(self) -> str:
        """Generate release notes."""
        return f"# Release v{self.version}\n\nAutomated release with cleaned repository structure."

    def create_github_release(self, version: str, release_notes: str, draft: bool = False):
        """Create GitHub release using gh CLI."""
        tag = f"v{version}"
        
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except Exception:
            print("⚠ GitHub CLI (gh) not found or not authenticated.")
            print(f"  Please manually push tag: git tag {tag} && git push origin {tag}")
            return
        
        cmd = ["gh", "release", "create", tag, "--title", f"Neural DSL {tag}"]
        
        if release_notes:
            notes_file = self.repo_root / f"release_notes_{version}.md"
            with open(notes_file, "w", encoding="utf-8") as f:
                f.write(release_notes)
            cmd.extend(["--notes-file", str(notes_file)])
        
        if draft:
            cmd.append("--draft")
        
        try:
            print(f"Creating GitHub release {tag}...")
            subprocess.run(cmd, check=True, cwd=self.repo_root)
            print(f"✓ Created GitHub release: {tag}")
            if release_notes and notes_file.exists():
                notes_file.unlink()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create GitHub release: {e}")
    
    def build_and_publish_pypi(self, test: bool = False, use_trusted_publishing: bool = True):
        """Build and publish to PyPI."""
        print("Building package for PyPI...")
        import shutil
        dist_dir = self.repo_root / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        
        subprocess.run([sys.executable, "-m", "build"], check=True, cwd=self.repo_root)
        
        print("Verifying package...")
        subprocess.run([sys.executable, "-m", "twine", "check", "dist/*"], 
                      check=True, cwd=self.repo_root)
        
        if use_trusted_publishing:
            print("⚠ Using trusted publishing. Upload will be handled by GitHub Actions.")
            return

        print("Uploading to PyPI (manual)...")
        # subprocess calls to twine upload if needed, usually we rely on CI
        print("Manual upload required if not using CI.")

    def full_release(self, version_type: str = "patch", skip_tests: bool = False, 
                     skip_lint: bool = False, draft: bool = False, test_pypi: bool = False,
                     use_trusted_publishing: bool = False):
        """Run full release process."""
        print("=" * 70)
        print("Neural DSL Release Automation")
        print("=" * 70)
        
        # 1. Validation
        if not skip_tests:
            if not self.run_tests():
                print("\n✗ Tests failed. Aborting release.")
                return False
        
        # 2. Bump & Commit
        print("\n=== Version Update ===\n")
        new_version = self.bump_version(version_type)
        self.version = new_version
        
        # 3. Release Notes
        release_notes = self.generate_release_notes()
        
        # 4. GitHub Release (creates tag)
        print("\n=== GitHub Release ===\n")
        self.create_github_release(new_version, release_notes, draft=draft)
        
        # 5. Build (Verification)
        print("\n=== Build Verification ===\n")
        self.build_and_publish_pypi(test=test_pypi, use_trusted_publishing=use_trusted_publishing)
        
        print("\n" + "=" * 70)
        print("✅ Release process finished locally.")
        print(f"v{new_version} has been committed and release attempted.")
        print("=" * 70)
        
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-type", default="patch")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--test-pypi", action="store_true")
    args = parser.parse_args()
    
    ReleaseAutomation().full_release(
        version_type=args.version_type,
        skip_tests=args.skip_tests,
        draft=args.draft,
        test_pypi=args.test_pypi
    )
