"""
Social Media Post Generator

Generates posts for Twitter/X, LinkedIn, Dev.to etc. from release information.
Includes API integration for automated posting.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests


class SocialMediaGenerator:
    """Generate and post social media content."""
    
    def __init__(self, version: str, release_notes: Dict):
        """Initialize generator."""
        self.version = version
        self.release_notes = release_notes
        self.dev_to_api_key = os.environ.get("DEV_TO_API_KEY")
        self.twitter_bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        self.twitter_api_key = os.environ.get("TWITTER_API_KEY")
        self.twitter_api_secret = os.environ.get("TWITTER_API_SECRET")
        self.twitter_access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
        self.twitter_access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
        self.linkedin_access_token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    
    def generate_twitter_post(self, max_length: int = 280) -> str:
        """Generate Twitter/X post (respects character limit)."""
        post = f"🚀 Neural DSL v{self.version} is here!\n\n"
        
        if self.release_notes.get("sections", {}).get("Added"):
            features = self.release_notes["sections"]["Added"][:2]
            for feature in features:
                feature_clean = feature.split('\n')[0]
                feature_clean = feature_clean.replace('**', '').strip()
                if feature_clean.startswith('**') and '**:' in feature_clean:
                    feature_clean = feature_clean.split('**:', 1)[-1].strip()
                feature_short = feature_clean[:80] + "..." if len(feature_clean) > 80 else feature_clean
                post += f"✨ {feature_short}\n"
        
        post += f"\n📦 pip install --upgrade neural-dsl\n"
        post += f"\n🔗 https://github.com/Lemniscate-SHA-256/Neural\n"
        post += f"\n#NeuralNetworks #Python #MachineLearning #DeepLearning #OpenSource"
        
        if len(post) > max_length:
            base = f"🚀 Neural DSL v{self.version} is here!\n\n"
            base += f"📦 pip install --upgrade neural-dsl\n"
            base += f"🔗 https://github.com/Lemniscate-SHA-256/Neural\n"
            base += f"\n#NeuralNetworks #Python #ML"
            post = base
        
        return post
    
    def generate_linkedin_post(self) -> str:
        """Generate LinkedIn post."""
        post = f"🎉 Exciting News: Neural DSL v{self.version} Release!\n\n"
        post += f"We're thrilled to announce the latest release of Neural DSL, making neural network development easier than ever.\n\n"
        
        if self.release_notes.get("sections", {}).get("Added"):
            post += "✨ What's New:\n\n"
            for feature in self.release_notes["sections"]["Added"][:5]:
                feature_clean = feature.split('\n')[0]
                feature_clean = feature_clean.replace('**', '').strip()
                if '**:' in feature_clean:
                    parts = feature_clean.split('**:', 1)
                    if len(parts) > 1:
                        feature_clean = f"{parts[0]}: {parts[1]}"
                post += f"• {feature_clean}\n"
            post += "\n"
        
        post += f"🚀 Get Started:\n\n"
        post += f"pip install --upgrade neural-dsl\n\n"
        post += f"📚 Learn more: https://github.com/Lemniscate-SHA-256/Neural\n\n"
        post += f"#NeuralNetworks #Python #MachineLearning #DeepLearning #AI #OpenSource #SoftwareDevelopment"
        
        return post
    
    def generate_devto_article(self, title: Optional[str] = None, 
                               published: bool = False) -> Dict:
        """Generate Dev.to article format."""
        if not title:
            title = f"Neural DSL v{self.version} Release - What's New"
        
        body_md = f"""# Neural DSL v{self.version} Release: What's New 🚀

We're thrilled to announce **Neural DSL v{self.version}**! This release includes exciting new features, improvements, and bug fixes.

"""
        
        sections = self.release_notes.get("sections", {})
        
        if sections.get("Added"):
            body_md += "## ✨ New Features\n\n"
            for item in sections["Added"]:
                body_md += f"- {item}\n\n"
        
        if sections.get("Improved"):
            body_md += "## 🚀 Improvements\n\n"
            for item in sections["Improved"]:
                body_md += f"- {item}\n\n"
        
        if sections.get("Fixed"):
            body_md += "## 🐛 Bug Fixes\n\n"
            for item in sections["Fixed"]:
                body_md += f"- {item}\n\n"
        
        body_md += """
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
        
        article = {
            "article": {
                "title": title,
                "published": published,
                "body_markdown": body_md,
                "tags": ["neuralnetworks", "python", "machinelearning", "deeplearning"],
                "canonical_url": "https://github.com/Lemniscate-SHA-256/Neural",
            }
        }
        
        return article
    
    def post_to_twitter(self, text: Optional[str] = None) -> Dict:
        """Post to Twitter/X using API v2."""
        if not text:
            text = self.generate_twitter_post()
        
        if not all([self.twitter_api_key, self.twitter_api_secret, 
                   self.twitter_access_token, self.twitter_access_token_secret]):
            return {"error": "Twitter API credentials not configured"}
        
        try:
            import tweepy
            
            client = tweepy.Client(
                bearer_token=self.twitter_bearer_token,
                consumer_key=self.twitter_api_key,
                consumer_secret=self.twitter_api_secret,
                access_token=self.twitter_access_token,
                access_token_secret=self.twitter_access_token_secret,
                wait_on_rate_limit=True
            )
            
            response = client.create_tweet(text=text)
            
            return {
                "success": True,
                "tweet_id": response.data['id'],
                "message": "Tweet posted successfully"
            }
            
        except ImportError:
            return {"error": "tweepy not installed. Install with: pip install tweepy"}
        except tweepy.TweepyException as e:
            return {"error": f"Twitter API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def post_to_devto(self, article: Optional[Dict] = None) -> Dict:
        """Post article to Dev.to using API."""
        if not self.dev_to_api_key:
            return {"error": "DEV_TO_API_KEY not configured"}
        
        if not article:
            article = self.generate_devto_article()
        
        headers = {
            "api-key": self.dev_to_api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://dev.to/api/articles",
                json=article,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                return {
                    "success": True,
                    "article_id": data.get("id"),
                    "url": data.get("url"),
                    "message": "Article posted to Dev.to successfully"
                }
            else:
                return {
                    "error": f"Dev.to API error: {response.status_code}",
                    "details": response.text
                }
        
        except requests.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def post_to_linkedin(self, text: Optional[str] = None) -> Dict:
        """Post to LinkedIn using API."""
        if not self.linkedin_access_token:
            return {"error": "LINKEDIN_ACCESS_TOKEN not configured"}
        
        if not text:
            text = self.generate_linkedin_post()
        
        headers = {
            "Authorization": f"Bearer {self.linkedin_access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        
        try:
            me_response = requests.get(
                "https://api.linkedin.com/v2/me",
                headers=headers,
                timeout=30
            )
            
            if me_response.status_code != 200:
                return {
                    "error": f"LinkedIn API error: {me_response.status_code}",
                    "details": me_response.text
                }
            
            author_id = me_response.json().get("id")
            
            post_data = {
                "author": f"urn:li:person:{author_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": text
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }
            
            response = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                json=post_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                return {
                    "success": True,
                    "post_id": data.get("id"),
                    "message": "Posted to LinkedIn successfully"
                }
            else:
                return {
                    "error": f"LinkedIn API error: {response.status_code}",
                    "details": response.text
                }
        
        except requests.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def save_posts(self, output_dir: str = "docs/social"):
        """Save social media posts to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        twitter_post = self.generate_twitter_post()
        twitter_path = os.path.join(output_dir, f"twitter_v{self.version}.txt")
        with open(twitter_path, "w", encoding="utf-8") as f:
            f.write(twitter_post)
        print(f"✓ Generated Twitter post: {twitter_path}")
        
        linkedin_post = self.generate_linkedin_post()
        linkedin_path = os.path.join(output_dir, f"linkedin_v{self.version}.txt")
        with open(linkedin_path, "w", encoding="utf-8") as f:
            f.write(linkedin_post)
        print(f"✓ Generated LinkedIn post: {linkedin_path}")
        
        return {
            "twitter": twitter_path,
            "linkedin": linkedin_path
        }
    
    def post_all(self, platforms: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Post to all configured platforms."""
        if not platforms:
            platforms = ["twitter", "devto", "linkedin"]
        
        results = {}
        
        if "twitter" in platforms:
            print("Posting to Twitter/X...")
            results["twitter"] = self.post_to_twitter()
            time.sleep(1)
        
        if "devto" in platforms:
            print("Posting to Dev.to...")
            results["devto"] = self.post_to_devto()
            time.sleep(1)
        
        if "linkedin" in platforms:
            print("Posting to LinkedIn...")
            results["linkedin"] = self.post_to_linkedin()
            time.sleep(1)
        
        return results


if __name__ == "__main__":
    import sys
    from blog_generator import BlogGenerator
    
    version = sys.argv[1] if len(sys.argv) > 1 else None
    
    generator = BlogGenerator(version=version)
    social_gen = SocialMediaGenerator(generator.version, generator.release_notes)
    
    print(f"Generating social media posts for version {generator.version}...")
    paths = social_gen.save_posts()
    
    print("\n✅ Social media posts generated successfully!")
    print(f"\nFiles created:")
    for platform, path in paths.items():
        print(f"  - {platform}: {path}")
    
    post_online = input("\nDo you want to post to social media platforms? (y/N): ").strip().lower()
    
    if post_online == 'y':
        platforms_input = input("Enter platforms (comma-separated: twitter,devto,linkedin) or press Enter for all: ").strip()
        platforms = [p.strip() for p in platforms_input.split(',')] if platforms_input else None
        
        print("\nPosting to platforms...")
        results = social_gen.post_all(platforms)
        
        print("\n📊 Posting Results:")
        for platform, result in results.items():
            if result.get("success"):
                print(f"  ✓ {platform}: {result.get('message')}")
            else:
                print(f"  ✗ {platform}: {result.get('error')}")
