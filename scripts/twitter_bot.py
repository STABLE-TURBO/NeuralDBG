"""
Twitter Bot for Neural DSL Release Announcements

Uses Twitter API v2 with proper rate limiting and error handling.
"""

import os
import re
import time
from typing import List, Optional


def parse_changelog(version: str) -> List[str]:
    """Extract changelog entries for a specific version from CHANGELOG.md"""
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            changelog = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("CHANGELOG.md not found")
    
    version_escaped = re.escape(version)
    version_pattern = rf"##\s*\[{version_escaped}\].*?\n(.*?)(?=\n##\s*\[|\Z)"
    match = re.search(version_pattern, changelog, re.DOTALL)
    
    if not match:
        version_pattern = rf"##\s*{version_escaped}.*?\n(.*?)(?=\n##\s*|\Z)"
        match = re.search(version_pattern, changelog, re.DOTALL)
    
    if not match:
        raise ValueError(f"Version {version} not found in CHANGELOG.md")
    
    content = match.group(1).strip()
    
    added_pattern = r"###\s*Added\s*\n(.*?)(?=\n###|\Z)"
    added_match = re.search(added_pattern, content, re.DOTALL)
    
    changes = []
    if added_match:
        section_text = added_match.group(1).strip()
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('- **'):
                feature_match = re.match(r'-\s*\*\*([^*]+)\*\*', line)
                if feature_match:
                    changes.append(feature_match.group(1).strip())
            elif line.startswith('- '):
                changes.append(line[2:].strip())
        
        return changes[:5]
    
    item_pattern = r'^-\s+(.+)$'
    items = re.findall(item_pattern, content, re.MULTILINE)
    return [item.strip() for item in items[:5]]


def format_tweet(version: str, changes: List[str], max_length: int = 280) -> str:
    """Create tweet text with version, changes, and truncate if needed"""
    if not changes:
        base_text = (
            f"🚀 Neural DSL v{version} released!\n\n"
            f"Check out what's new!\n\n"
            f"#MachineLearning #Python #NeuralNetworks\n"
            f"GitHub: https://github.com/Lemniscate-SHA-256/Neural/releases/tag/v{version}"
        )
        return base_text if len(base_text) <= max_length else base_text[:max_length-3] + "..."
    
    change_lines = []
    for change in changes[:3]:
        change_clean = change.split('\n')[0]
        change_clean = change_clean.replace('**', '').strip()
        if len(change_clean) > 60:
            change_clean = change_clean[:57] + "..."
        change_lines.append(f"• {change_clean}")
    
    change_text = "\n".join(change_lines)
    
    base_text = (
        f"🚀 Neural DSL v{version} released!\n"
        f"{change_text}\n\n"
        f"#MachineLearning #Python\n"
        f"GitHub: https://github.com/Lemniscate-SHA-256/Neural/releases/tag/v{version}"
    )
    
    if len(base_text) > max_length:
        change_lines = []
        for change in changes[:2]:
            change_clean = change.split('\n')[0].replace('**', '').strip()
            if len(change_clean) > 50:
                change_clean = change_clean[:47] + "..."
            change_lines.append(f"• {change_clean}")
        
        change_text = "\n".join(change_lines)
        base_text = (
            f"🚀 Neural DSL v{version}!\n"
            f"{change_text}\n\n"
            f"#ML #Python\n"
            f"https://github.com/Lemniscate-SHA-256/Neural/releases/tag/v{version}"
        )
    
    if len(base_text) > max_length:
        base_text = base_text[:max_length-3] + "..."
    
    return base_text


def post_release(version: str, dry_run: bool = False) -> dict:
    """
    Post a release announcement to Twitter/X.
    
    Args:
        version: Version number to announce
        dry_run: If True, only print the tweet without posting
    
    Returns:
        dict with status and message/error
    """
    try:
        import tweepy
    except ImportError:
        return {
            "success": False,
            "error": "tweepy not installed. Install with: pip install tweepy"
        }
    
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    
    if not all([api_key, api_secret, access_token, access_token_secret]):
        return {
            "success": False,
            "error": "Twitter API credentials not configured. Set TWITTER_API_KEY, "
                    "TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, and TWITTER_ACCESS_TOKEN_SECRET"
        }
    
    try:
        changes = parse_changelog(version)
        tweet_text = format_tweet(version, changes)
        
        if dry_run:
            print("=" * 60)
            print("DRY RUN - Tweet would be:")
            print("=" * 60)
            print(tweet_text)
            print("=" * 60)
            print(f"Length: {len(tweet_text)} characters")
            return {
                "success": True,
                "message": "Dry run completed",
                "tweet": tweet_text
            }
        
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True
        )
        
        response = client.create_tweet(text=tweet_text)
        
        tweet_id = response.data['id']
        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
        
        return {
            "success": True,
            "message": "Tweet posted successfully!",
            "tweet_id": tweet_id,
            "tweet_url": tweet_url,
            "tweet": tweet_text
        }
    
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except tweepy.TooManyRequests as e:
        retry_after = getattr(e, 'retry_after', 900)
        return {
            "success": False,
            "error": f"Rate limit exceeded. Retry after {retry_after} seconds",
            "retry_after": retry_after
        }
    except tweepy.Forbidden as e:
        return {
            "success": False,
            "error": f"Twitter API Forbidden error: {str(e)}. Check your API permissions."
        }
    except tweepy.Unauthorized as e:
        return {
            "success": False,
            "error": f"Twitter API Unauthorized: {str(e)}. Check your credentials."
        }
    except tweepy.TweepyException as e:
        return {"success": False, "error": f"Twitter API error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def post_release_with_retry(version: str, max_retries: int = 3, 
                            initial_delay: int = 60, dry_run: bool = False) -> dict:
    """
    Post release with exponential backoff retry logic.
    
    Args:
        version: Version number to announce
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retry
        dry_run: If True, only print the tweet without posting
    
    Returns:
        dict with status and message/error
    """
    for attempt in range(max_retries):
        result = post_release(version, dry_run=dry_run)
        
        if result["success"]:
            return result
        
        if "retry_after" in result:
            wait_time = result["retry_after"]
            print(f"Rate limited. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        elif attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            return result
    
    return {"success": False, "error": "Max retries exceeded"}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python twitter_bot.py <version> [--dry-run]")
        print("Example: python twitter_bot.py 0.3.0")
        print("         python twitter_bot.py 0.3.0 --dry-run")
        sys.exit(1)
    
    version = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    
    print(f"Posting release announcement for v{version}...")
    if dry_run:
        print("(Dry run mode - will not actually post)")
    
    result = post_release_with_retry(version, dry_run=dry_run)
    
    if result["success"]:
        print(f"✓ {result['message']}")
        if "tweet_url" in result:
            print(f"  Tweet URL: {result['tweet_url']}")
    else:
        print(f"✗ Error: {result['error']}")
        sys.exit(1)
