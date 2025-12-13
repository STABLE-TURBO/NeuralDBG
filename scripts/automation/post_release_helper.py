"""
Post-Release Automation Helper

Helper functions for post-release automation workflow.
Handles GitHub Discussions, issue creation, and notifications.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests


class PostReleaseHelper:
    """Helper class for post-release automation tasks."""
    
    def __init__(self, github_token: str, repository: str):
        """
        Initialize post-release helper.
        
        Args:
            github_token: GitHub API token
            repository: Repository in format "owner/repo"
        """
        self.github_token = github_token
        self.repository = repository
        self.api_base = "https://api.github.com"
        self.graphql_url = "https://api.github.com/graphql"
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def extract_release_notes(self, version: str, changelog_path: str = "CHANGELOG.md") -> str:
        """
        Extract release notes for a specific version from CHANGELOG.
        
        Args:
            version: Version to extract notes for (e.g., "0.3.0")
            changelog_path: Path to CHANGELOG.md
            
        Returns:
            Release notes text
        """
        try:
            with open(changelog_path, 'r', encoding='utf-8') as f:
                changelog = f.read()
            
            # Match version section (handles both [X.Y.Z] and [X.Y.Z-dev] formats)
            pattern = rf'## \[{re.escape(version)}[^\]]*\].*?(?=\n## \[|\Z)'
            match = re.search(pattern, changelog, re.DOTALL)
            
            if match:
                notes = match.group(0).strip()
                # Remove the version header line
                notes = '\n'.join(notes.split('\n')[1:]).strip()
                return notes
            else:
                return f"Release {version} is now available! Check the CHANGELOG for details."
        except Exception as e:
            print(f"Warning: Could not extract release notes: {e}")
            return f"Release {version} is now available!"
    
    def get_repository_id(self) -> str:
        """
        Get repository ID using GraphQL.
        
        Returns:
            Repository node ID
        """
        query = """
        query($owner: String!, $name: String!) {
            repository(owner: $owner, name: $name) {
                id
            }
        }
        """
        
        owner, name = self.repository.split('/')
        variables = {"owner": owner, "name": name}
        
        response = requests.post(
            self.graphql_url,
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        response.raise_for_status()
        
        data = response.json()
        return data['data']['repository']['id']
    
    def get_discussion_category_id(self, category_name: str = "Announcements") -> Optional[str]:
        """
        Get discussion category ID by name.
        
        Args:
            category_name: Name of the discussion category
            
        Returns:
            Category node ID or None if not found
        """
        query = """
        query($owner: String!, $name: String!) {
            repository(owner: $owner, name: $name) {
                discussionCategories(first: 10) {
                    nodes {
                        id
                        name
                    }
                }
            }
        }
        """
        
        owner, name = self.repository.split('/')
        variables = {"owner": owner, "name": name}
        
        response = requests.post(
            self.graphql_url,
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        response.raise_for_status()
        
        data = response.json()
        categories = data['data']['repository']['discussionCategories']['nodes']
        
        for category in categories:
            if category['name'] == category_name:
                return category['id']
        
        # Try "General" as fallback
        for category in categories:
            if category['name'] == "General":
                return category['id']
        
        # Return first category as last resort
        if categories:
            return categories[0]['id']
        
        return None
    
    def create_discussion(self, version: str, release_notes: str) -> Tuple[bool, str]:
        """
        Create a GitHub Discussion for the release announcement.
        
        Args:
            version: Version that was released
            release_notes: Release notes content
            
        Returns:
            Tuple of (success: bool, discussion_url: str)
        """
        try:
            repo_id = self.get_repository_id()
            category_id = self.get_discussion_category_id()
            
            if not category_id:
                return False, "Could not find discussion category"
            
            title = f"🎉 Neural DSL v{version} Released!"
            body = f"""{release_notes}

---

**Installation:**
```bash
pip install --upgrade neural-dsl
```

**PyPI:** https://pypi.org/project/neural-dsl/{version}/
**GitHub Release:** https://github.com/{self.repository}/releases/tag/v{version}

We'd love to hear your feedback! Please share your thoughts, questions, or issues below.
"""
            
            mutation = """
            mutation($repositoryId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
                createDiscussion(input: {
                    repositoryId: $repositoryId,
                    categoryId: $categoryId,
                    title: $title,
                    body: $body
                }) {
                    discussion {
                        url
                    }
                }
            }
            """
            
            variables = {
                "repositoryId": repo_id,
                "categoryId": category_id,
                "title": title,
                "body": body
            }
            
            response = requests.post(
                self.graphql_url,
                json={"query": mutation, "variables": variables},
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'errors' in data:
                return False, f"GraphQL errors: {data['errors']}"
            
            discussion_url = data['data']['createDiscussion']['discussion']['url']
            return True, discussion_url
        
        except Exception as e:
            return False, str(e)
    
    def create_planning_issue(self, current_version: str, next_version: str) -> Tuple[bool, str]:
        """
        Create a planning issue for the next release.
        
        Args:
            current_version: Version that was just released
            next_version: Next planned version
            
        Returns:
            Tuple of (success: bool, issue_url: str)
        """
        try:
            # Calculate suggested next release date (5 weeks from now)
            next_date = datetime.now() + timedelta(weeks=5)
            date_str = next_date.strftime('%Y-%m-%d')
            
            title = f"📋 Planning: Release v{next_version}"
            body = f"""## Next Release Planning

This issue tracks planning for the next release of Neural DSL.

### Release Information
- **Previous Release:** v{current_version}
- **Target Version:** v{next_version}
- **Suggested Date:** {date_str} (approximately 5 weeks from now)

### Pre-Release Checklist
- [ ] Review and close completed issues
- [ ] Update CHANGELOG.md with all changes
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Review and update dependencies
- [ ] Test installation on clean environments
- [ ] Prepare release notes
- [ ] Create release candidate (if needed)

### Areas to Consider
- [ ] Bug fixes from user reports
- [ ] Performance improvements
- [ ] New features from roadmap
- [ ] Documentation improvements
- [ ] Dependency updates
- [ ] Security patches

### Discussion Topics
- What features should be prioritized?
- Are there any breaking changes needed?
- What user feedback should be addressed?

---

**Note:** This issue was automatically created after the v{current_version} release.
Feel free to adjust the timeline, add tasks, or close if not needed.
"""
            
            url = f"{self.api_base}/repos/{self.repository}/issues"
            payload = {
                "title": title,
                "body": body,
                "labels": ["release", "planning"]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return True, data['html_url']
        
        except Exception as e:
            return False, str(e)
    
    def send_discord_notification(self, version: str, webhook_url: str) -> bool:
        """
        Send release notification to Discord.
        
        Args:
            version: Version that was released
            webhook_url: Discord webhook URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            release_url = f"https://github.com/{self.repository}/releases/tag/v{version}"
            pypi_url = f"https://pypi.org/project/neural-dsl/{version}/"
            
            payload = {
                "embeds": [{
                    "title": f"🎉 Neural DSL v{version} Released!",
                    "description": "A new version of Neural DSL has been released.",
                    "color": 3066993,  # Green color
                    "fields": [
                        {
                            "name": "📦 Version",
                            "value": version,
                            "inline": True
                        },
                        {
                            "name": "🔗 Links",
                            "value": f"[GitHub Release]({release_url}) • [PyPI]({pypi_url})",
                            "inline": False
                        },
                        {
                            "name": "📥 Install",
                            "value": "```bash\npip install --upgrade neural-dsl\n```",
                            "inline": False
                        }
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            return True
        
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")
            return False
    
    def trigger_deployment(self, webhook_url: str, platform: str = "netlify") -> bool:
        """
        Trigger deployment webhook.
        
        Args:
            webhook_url: Deployment webhook URL
            platform: Platform name (netlify, vercel, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if platform.lower() == "netlify":
                response = requests.post(webhook_url, json={})
            elif platform.lower() == "vercel":
                response = requests.post(webhook_url)
            else:
                response = requests.post(webhook_url)
            
            response.raise_for_status()
            return True
        
        except Exception as e:
            print(f"Failed to trigger {platform} deployment: {e}")
            return False


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-release automation helper")
    parser.add_argument("--action", required=True, 
                       choices=["discussion", "issue", "discord", "deploy"],
                       help="Action to perform")
    parser.add_argument("--version", required=True, help="Released version")
    parser.add_argument("--next-version", help="Next version (for planning issue)")
    parser.add_argument("--webhook-url", help="Webhook URL (for discord/deploy)")
    parser.add_argument("--platform", default="netlify", help="Deployment platform")
    
    args = parser.parse_args()
    
    # Get environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    repository = os.environ.get('GITHUB_REPOSITORY')
    
    if not github_token or not repository:
        print("Error: GITHUB_TOKEN and GITHUB_REPOSITORY must be set")
        sys.exit(1)
    
    helper = PostReleaseHelper(github_token, repository)
    
    if args.action == "discussion":
        release_notes = helper.extract_release_notes(args.version)
        success, result = helper.create_discussion(args.version, release_notes)
        if success:
            print(f"✓ Created discussion: {result}")
        else:
            print(f"✗ Failed to create discussion: {result}")
            sys.exit(1)
    
    elif args.action == "issue":
        if not args.next_version:
            print("Error: --next-version required for issue creation")
            sys.exit(1)
        success, result = helper.create_planning_issue(args.version, args.next_version)
        if success:
            print(f"✓ Created issue: {result}")
        else:
            print(f"✗ Failed to create issue: {result}")
            sys.exit(1)
    
    elif args.action == "discord":
        if not args.webhook_url:
            print("Error: --webhook-url required for discord notification")
            sys.exit(1)
        success = helper.send_discord_notification(args.version, args.webhook_url)
        if success:
            print("✓ Sent Discord notification")
        else:
            print("✗ Failed to send Discord notification")
            sys.exit(1)
    
    elif args.action == "deploy":
        if not args.webhook_url:
            print("Error: --webhook-url required for deployment")
            sys.exit(1)
        success = helper.trigger_deployment(args.webhook_url, args.platform)
        if success:
            print(f"✓ Triggered {args.platform} deployment")
        else:
            print(f"✗ Failed to trigger {args.platform} deployment")
            sys.exit(1)


if __name__ == "__main__":
    main()
