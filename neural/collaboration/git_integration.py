"""
Git Integration - Version control integration for collaborative workspaces.

Provides Git repository operations for workspace files including commits,
branches, and synchronization.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neural.exceptions import SyncError


class GitIntegration:
    """
    Git integration for workspace version control.
    
    Provides Git operations for tracking changes in collaborative workspaces.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize Git integration.
        
        Parameters
        ----------
        repo_path : Path
            Path to Git repository
        """
        self.repo_path = Path(repo_path)
        
        if not self._is_git_available():
            raise SyncError("Git is not available. Please install Git.")
    
    def _is_git_available(self) -> bool:
        """Check if Git is available."""
        try:
            subprocess.run(
                ['git', '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _run_git_command(self, args: List[str], check: bool = True) -> Tuple[str, str]:
        """
        Run a Git command.
        
        Parameters
        ----------
        args : List[str]
            Git command arguments
        check : bool
            Whether to check return code
            
        Returns
        -------
        Tuple[str, str]
            stdout and stderr
        """
        result = subprocess.run(
            ['git'] + args,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check
        )
        return result.stdout, result.stderr
    
    def init_repo(self) -> bool:
        """
        Initialize a Git repository.
        
        Returns
        -------
        bool
            True if successful
        """
        try:
            if (self.repo_path / '.git').exists():
                return True
            
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self._run_git_command(['init'])
            
            gitignore_path = self.repo_path / '.gitignore'
            if not gitignore_path.exists():
                with open(gitignore_path, 'w') as f:
                    f.write("*.pyc\n__pycache__/\n.DS_Store\n")
            
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to initialize Git repository: {e.stderr}")
    
    def is_repo(self) -> bool:
        """
        Check if directory is a Git repository.
        
        Returns
        -------
        bool
            True if is a Git repository
        """
        return (self.repo_path / '.git').exists()
    
    def add_files(self, patterns: List[str]) -> bool:
        """
        Add files to Git staging area.
        
        Parameters
        ----------
        patterns : List[str]
            File patterns to add
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            for pattern in patterns:
                self._run_git_command(['add', pattern])
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to add files: {e.stderr}")
    
    def commit(
        self,
        message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None
    ) -> str:
        """
        Create a Git commit.
        
        Parameters
        ----------
        message : str
            Commit message
        author_name : Optional[str]
            Author name
        author_email : Optional[str]
            Author email
            
        Returns
        -------
        str
            Commit hash
        """
        try:
            args = ['commit', '-m', message]
            
            if author_name and author_email:
                args.extend(['--author', f'{author_name} <{author_email}>'])
            
            self._run_git_command(args)
            
            stdout, _ = self._run_git_command(['rev-parse', 'HEAD'])
            return stdout.strip()
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to create commit: {e.stderr}")
    
    def get_status(self) -> Dict[str, List[str]]:
        """
        Get repository status.
        
        Returns
        -------
        Dict[str, List[str]]
            Status with modified, staged, and untracked files
        """
        try:
            stdout, _ = self._run_git_command(['status', '--porcelain'])
            
            modified = []
            staged = []
            untracked = []
            
            for line in stdout.splitlines():
                if not line:
                    continue
                
                status = line[:2]
                filename = line[3:]
                
                if status[0] == 'M' or status[1] == 'M':
                    modified.append(filename)
                if status[0] in ['A', 'M', 'D']:
                    staged.append(filename)
                if status == '??':
                    untracked.append(filename)
            
            return {
                'modified': modified,
                'staged': staged,
                'untracked': untracked
            }
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to get status: {e.stderr}")
    
    def create_branch(self, branch_name: str) -> bool:
        """
        Create a new branch.
        
        Parameters
        ----------
        branch_name : str
            Branch name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            self._run_git_command(['branch', branch_name])
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to create branch: {e.stderr}")
    
    def checkout_branch(self, branch_name: str) -> bool:
        """
        Checkout a branch.
        
        Parameters
        ----------
        branch_name : str
            Branch name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            self._run_git_command(['checkout', branch_name])
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to checkout branch: {e.stderr}")
    
    def get_current_branch(self) -> str:
        """
        Get current branch name.
        
        Returns
        -------
        str
            Current branch name
        """
        try:
            stdout, _ = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
            return stdout.strip()
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to get current branch: {e.stderr}")
    
    def list_branches(self) -> List[str]:
        """
        List all branches.
        
        Returns
        -------
        List[str]
            List of branch names
        """
        try:
            stdout, _ = self._run_git_command(['branch', '--list'])
            branches = []
            for line in stdout.splitlines():
                branch = line.strip().lstrip('* ')
                if branch:
                    branches.append(branch)
            return branches
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to list branches: {e.stderr}")
    
    def merge_branch(
        self,
        branch_name: str,
        strategy: Optional[str] = None
    ) -> bool:
        """
        Merge a branch into current branch.
        
        Parameters
        ----------
        branch_name : str
            Branch to merge
        strategy : Optional[str]
            Merge strategy
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            args = ['merge', branch_name]
            if strategy:
                args.extend(['--strategy', strategy])
            
            self._run_git_command(args)
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to merge branch: {e.stderr}")
    
    def get_log(self, n: int = 10) -> List[Dict]:
        """
        Get commit log.
        
        Parameters
        ----------
        n : int
            Number of commits to retrieve
            
        Returns
        -------
        List[Dict]
            List of commits with hash, author, date, and message
        """
        try:
            stdout, _ = self._run_git_command([
                'log',
                f'-n{n}',
                '--pretty=format:%H|%an|%ae|%ai|%s'
            ])
            
            commits = []
            for line in stdout.splitlines():
                if not line:
                    continue
                
                parts = line.split('|', 4)
                if len(parts) == 5:
                    commits.append({
                        'hash': parts[0],
                        'author_name': parts[1],
                        'author_email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    })
            
            return commits
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to get log: {e.stderr}")
    
    def diff(self, ref1: Optional[str] = None, ref2: Optional[str] = None) -> str:
        """
        Get diff between refs or working directory.
        
        Parameters
        ----------
        ref1 : Optional[str]
            First ref (e.g., commit hash, branch)
        ref2 : Optional[str]
            Second ref
            
        Returns
        -------
        str
            Diff output
        """
        try:
            args = ['diff']
            if ref1:
                args.append(ref1)
            if ref2:
                args.append(ref2)
            
            stdout, _ = self._run_git_command(args)
            return stdout
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to get diff: {e.stderr}")
    
    def pull(self, remote: str = 'origin', branch: Optional[str] = None) -> bool:
        """
        Pull changes from remote.
        
        Parameters
        ----------
        remote : str
            Remote name
        branch : Optional[str]
            Branch name
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            args = ['pull', remote]
            if branch:
                args.append(branch)
            
            self._run_git_command(args)
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to pull: {e.stderr}")
    
    def push(
        self,
        remote: str = 'origin',
        branch: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Push changes to remote.
        
        Parameters
        ----------
        remote : str
            Remote name
        branch : Optional[str]
            Branch name
        force : bool
            Force push
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            args = ['push', remote]
            if branch:
                args.append(branch)
            if force:
                args.append('--force')
            
            self._run_git_command(args)
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to push: {e.stderr}")
    
    def add_remote(self, name: str, url: str) -> bool:
        """
        Add a remote repository.
        
        Parameters
        ----------
        name : str
            Remote name
        url : str
            Remote URL
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            self._run_git_command(['remote', 'add', name, url])
            return True
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to add remote: {e.stderr}")
    
    def list_remotes(self) -> Dict[str, str]:
        """
        List remote repositories.
        
        Returns
        -------
        Dict[str, str]
            Mapping of remote names to URLs
        """
        try:
            stdout, _ = self._run_git_command(['remote', '-v'])
            remotes = {}
            
            for line in stdout.splitlines():
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    if name not in remotes:
                        remotes[name] = url
            
            return remotes
        except subprocess.CalledProcessError as e:
            raise SyncError(f"Failed to list remotes: {e.stderr}")
