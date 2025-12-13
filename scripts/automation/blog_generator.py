"""
Automated Blog Post Generator

Generates blog posts from CHANGELOG.md and version information.
Supports multiple platforms: Medium, Dev.to, Hashnode, etc.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BlogGenerator:
    """Generate blog posts from changelog and version info."""
    
    def __init__(self, changelog_path: str = "CHANGELOG.md", version: Optional[str] = None):
        """
        Initialize blog generator.
        
        Args:
            changelog_path: Path to CHANGELOG.md
            version: Version number (auto-detected if None)
        """
        self.changelog_path = changelog_path
        self.version = version or self._detect_version()
        self.changelog_content = self._read_changelog()
        self.release_notes = self._extract_release_notes()
    
    def _detect_version(self) -> str:
        """Detect current version from setup.py or __init__.py."""
        try:
            with open("setup.py", "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
        try:
            with open("neural/__init__.py", "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
        return "0.3.0-dev"
    
    def _read_changelog(self) -> str:
        """Read changelog file."""
        try:
            with open(self.changelog_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading changelog: {e}")
            return ""
    
    def _extract_release_notes(self) -> Dict[str, any]:
        """Extract release notes for current version."""
        if not self.changelog_content:
            return {"content": "", "date": datetime.now().strftime("%Y-%m-%d"), "sections": {}}
        
        version_escaped = re.escape(self.version)
        version_pattern = rf"##\s*\[{version_escaped}\].*?\n(.*?)(?=\n##\s*\[|\Z)"
        match = re.search(version_pattern, self.changelog_content, re.DOTALL)
        
        if not match:
            version_pattern = rf"##\s*{version_escaped}.*?\n(.*?)(?=\n##\s*|\Z)"
            match = re.search(version_pattern, self.changelog_content, re.DOTALL)
        
        if not match:
            return {"content": "", "date": datetime.now().strftime("%Y-%m-%d"), "sections": {}}
        
        content = match.group(1).strip()
        
        date_match = re.search(r"\[[\d\w.-]+\]\s*-\s*(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", self.changelog_content)
        if not date_match:
            date_match = re.search(r"##[^-]+-\s*(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", self.changelog_content)
        date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
        
        sections = {
            "Added": self._extract_section(content, "### Added"),
            "Fixed": self._extract_section(content, "### Fixed"),
            "Changed": self._extract_section(content, "### Changed"),
            "Improved": self._extract_section(content, "### Improved"),
            "Removed": self._extract_section(content, "### Removed"),
            "Deprecated": self._extract_section(content, "### Deprecated"),
            "Security": self._extract_section(content, "### Security"),
            "Technical": self._extract_section(content, "### Technical"),
        }
        
        return {
            "content": content,
            "date": date,
            "sections": {k: v for k, v in sections.items() if v},
            "full_text": content
        }
    
    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """Extract items from a changelog section."""
        pattern = rf"{re.escape(section_name)}\s*\n(.*?)(?=\n###|\n---|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return []
        
        section_text = match.group(1).strip()
        items = []
        
        lines = section_text.split('\n')
        current_item = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('- **'):
                if current_item:
                    items.append(current_item)
                feature_match = re.match(r'-\s*\*\*([^*]+)\*\*:?\s*(.*)', line)
                if feature_match:
                    title = feature_match.group(1).strip()
                    desc = feature_match.group(2).strip()
                    current_item = f"**{title}**: {desc}" if desc else f"**{title}**"
                else:
                    current_item = line[2:].strip()
            elif line.startswith('- '):
                if current_item:
                    items.append(current_item)
                current_item = line[2:].strip()
            elif current_item and line.startswith('  - '):
                sub_item = line[4:].strip()
                current_item += f"\n  - {sub_item}"
            elif current_item:
                current_item += f" {line}"
        
        if current_item:
            items.append(current_item)
        
        return items
    
    def generate_medium_post(self) -> str:
        """Generate blog post for Medium."""
        sections = self.release_notes.get("sections", {})
        has_content = any(sections.values())
        
        if not has_content:
            return f"""# Neural DSL v{self.version} Release

*Published on {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}*

We're excited to announce the release of **Neural DSL v{self.version}**!

## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 🔗 Resources

- **GitHub**: https://github.com/Lemniscate-SHA-256/Neural
- **Documentation**: https://github.com/Lemniscate-SHA-256/Neural#readme
- **PyPI**: https://pypi.org/project/neural-dsl/

---

*For the full changelog, visit our [GitHub repository](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md).*
"""
        
        template = f"""# Neural DSL v{self.version} Release: What's New

*Published on {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}*

We're excited to announce the release of **Neural DSL v{self.version}**! This release brings significant improvements, new features, and bug fixes to make neural network development easier than ever.

## 🎉 What's New

"""
        
        if sections.get("Added"):
            template += "### ✨ New Features\n\n"
            for item in sections["Added"]:
                template += f"- {item}\n\n"
        
        if sections.get("Improved"):
            template += "\n### 🚀 Improvements\n\n"
            for item in sections["Improved"]:
                template += f"- {item}\n\n"
        
        if sections.get("Fixed"):
            template += "\n### 🐛 Bug Fixes\n\n"
            for item in sections["Fixed"]:
                template += f"- {item}\n\n"
        
        if sections.get("Changed"):
            template += "\n### 🔄 Changes\n\n"
            for item in sections["Changed"]:
                template += f"- {item}\n\n"
        
        if sections.get("Technical"):
            template += "\n### ⚙️ Technical\n\n"
            for item in sections["Technical"]:
                template += f"- {item}\n\n"
        
        template += f"""
## 📦 Installation

Get the latest version:

```bash
pip install --upgrade neural-dsl
```

## 🔗 Resources

- **GitHub**: https://github.com/Lemniscate-SHA-256/Neural
- **Documentation**: https://github.com/Lemniscate-SHA-256/Neural#readme
- **PyPI**: https://pypi.org/project/neural-dsl/

## 🙏 Thank You

Thank you to all contributors and users who helped make this release possible!

---

*For the full changelog, visit our [GitHub repository](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md).*
"""
        
        return template
    
    def generate_devto_post(self) -> str:
        """Generate blog post for Dev.to with proper frontmatter."""
        sections = self.release_notes.get("sections", {})
        has_content = any(sections.values())
        
        if not has_content:
            return f"""---
title: Neural DSL v{self.version} Release
published: false
description: Announcing Neural DSL v{self.version}
tags: neuralnetworks, python, machinelearning, deeplearning
canonical_url: https://github.com/Lemniscate-SHA-256/Neural
---

# Neural DSL v{self.version} Release 🚀

We're thrilled to announce **Neural DSL v{self.version}**!

## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 🔗 Links

- [GitHub](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural#readme)
- [PyPI](https://pypi.org/project/neural-dsl/)

---

*Full changelog: [GitHub](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md)*
"""
        
        template = f"""---
title: Neural DSL v{self.version} Release - What's New
published: false
description: Announcing Neural DSL v{self.version} with new features, improvements, and bug fixes
tags: neuralnetworks, python, machinelearning, deeplearning
canonical_url: https://github.com/Lemniscate-SHA-256/Neural
cover_image: https://raw.githubusercontent.com/Lemniscate-SHA-256/Neural/main/docs/images/neural-banner.png
---

# Neural DSL v{self.version} Release: What's New 🚀

We're thrilled to announce **Neural DSL v{self.version}**! This release includes exciting new features, improvements, and bug fixes.

"""
        
        if sections.get("Added"):
            template += "## ✨ New Features\n\n"
            for item in sections["Added"]:
                template += f"- {item}\n\n"
        
        if sections.get("Improved"):
            template += "## 🚀 Improvements\n\n"
            for item in sections["Improved"]:
                template += f"- {item}\n\n"
        
        if sections.get("Fixed"):
            template += "## 🐛 Bug Fixes\n\n"
            for item in sections["Fixed"]:
                template += f"- {item}\n\n"
        
        if sections.get("Changed"):
            template += "## 🔄 Changes\n\n"
            for item in sections["Changed"]:
                template += f"- {item}\n\n"
        
        if sections.get("Technical"):
            template += "## ⚙️ Technical Updates\n\n"
            for item in sections["Technical"]:
                template += f"- {item}\n\n"
        
        template += f"""
## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 🔗 Links

- [GitHub](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural#readme)
- [PyPI](https://pypi.org/project/neural-dsl/)

---

*Full changelog: [GitHub](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md)*
"""
        
        return template
    
    def generate_hashnode_post(self) -> str:
        """Generate blog post for Hashnode."""
        sections = self.release_notes.get("sections", {})
        has_content = any(sections.values())
        
        template = f"""---
title: "Neural DSL v{self.version} Release: What's New"
subtitle: "Announcing Neural DSL v{self.version} with new features and improvements"
slug: neural-dsl-v{self.version.replace('.', '-')}-release
tags: neural-networks, python, machine-learning, deep-learning
domain: your-hashnode-blog.hashnode.dev
publishAs: draft
---

# Neural DSL v{self.version} Release: What's New 🚀

*Published on {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}*

We're excited to announce **Neural DSL v{self.version}**!

"""
        
        if has_content:
            if sections.get("Added"):
                template += "## ✨ New Features\n\n"
                for item in sections["Added"]:
                    template += f"- {item}\n\n"
            
            if sections.get("Improved"):
                template += "## 🚀 Improvements\n\n"
                for item in sections["Improved"]:
                    template += f"- {item}\n\n"
            
            if sections.get("Fixed"):
                template += "## 🐛 Bug Fixes\n\n"
                for item in sections["Fixed"]:
                    template += f"- {item}\n\n"
        
        template += f"""
## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 🔗 Resources

- [GitHub Repository](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural#readme)
- [PyPI Package](https://pypi.org/project/neural-dsl/)
- [Full Changelog](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md)

Thank you for using Neural DSL! 🙏
"""
        
        return template
    
    def generate_github_release_notes(self) -> str:
        """Generate GitHub release notes."""
        sections = self.release_notes.get("sections", {})
        has_content = any(sections.values())
        
        template = f"""# Neural DSL v{self.version}

Release date: {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}

"""
        
        if has_content:
            if sections.get("Added"):
                template += "## ✨ What's New\n\n"
                for item in sections["Added"]:
                    template += f"- {item}\n"
                template += "\n"
            
            if sections.get("Improved"):
                template += "## 🚀 Improvements\n\n"
                for item in sections["Improved"]:
                    template += f"- {item}\n"
                template += "\n"
            
            if sections.get("Fixed"):
                template += "## 🐛 Bug Fixes\n\n"
                for item in sections["Fixed"]:
                    template += f"- {item}\n"
                template += "\n"
            
            if sections.get("Changed"):
                template += "## 🔄 Changes\n\n"
                for item in sections["Changed"]:
                    template += f"- {item}\n"
                template += "\n"
            
            if sections.get("Technical"):
                template += "## ⚙️ Technical Updates\n\n"
                for item in sections["Technical"]:
                    template += f"- {item}\n"
                template += "\n"
        
        template += f"""## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 📚 Documentation

- [Full Changelog](CHANGELOG.md)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural#readme)
- [Examples](examples/)

## 🙏 Contributors

Thank you to everyone who contributed to this release!
"""
        
        return template
    
    def save_blog_posts(self, output_dir: str = "docs/blog"):
        """Save generated blog posts to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        medium_post = self.generate_medium_post()
        medium_path = os.path.join(output_dir, f"medium_v{self.version}_release.md")
        with open(medium_path, "w", encoding="utf-8") as f:
            f.write(medium_post)
        print(f"✓ Generated Medium post: {medium_path}")
        
        devto_post = self.generate_devto_post()
        devto_path = os.path.join(output_dir, f"devto_v{self.version}_release.md")
        with open(devto_path, "w", encoding="utf-8") as f:
            f.write(devto_post)
        print(f"✓ Generated Dev.to post: {devto_path}")
        
        hashnode_post = self.generate_hashnode_post()
        hashnode_path = os.path.join(output_dir, f"hashnode_v{self.version}_release.md")
        with open(hashnode_path, "w", encoding="utf-8") as f:
            f.write(hashnode_post)
        print(f"✓ Generated Hashnode post: {hashnode_path}")
        
        github_notes = self.generate_github_release_notes()
        github_path = os.path.join(output_dir, f"github_v{self.version}_release.md")
        with open(github_path, "w", encoding="utf-8") as f:
            f.write(github_notes)
        print(f"✓ Generated GitHub release notes: {github_path}")
        
        return {
            "medium": medium_path,
            "devto": devto_path,
            "hashnode": hashnode_path,
            "github": github_path
        }


if __name__ == "__main__":
    import sys
    
    version = sys.argv[1] if len(sys.argv) > 1 else None
    generator = BlogGenerator(version=version)
    
    print(f"Generating blog posts for version {generator.version}...")
    paths = generator.save_blog_posts()
    
    print("\n✅ Blog posts generated successfully!")
    print(f"\nFiles created:")
    for platform, path in paths.items():
        print(f"  - {platform}: {path}")
