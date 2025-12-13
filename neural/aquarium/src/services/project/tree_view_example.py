from pathlib import Path
from typing import List
from .file_tree import FileTree
from .file_node import FileNode, FileNodeType


class TreeViewRenderer:
    def __init__(self):
        self.indent_size = 2
        self.dir_icon = "📁"
        self.file_icon = "📄"
        self.neural_icon = "🧠"
        self.expanded_icon = "▼"
        self.collapsed_icon = "▶"
        
    def render_tree(self, tree: FileTree) -> str:
        if not tree.root:
            return "Empty tree"
        
        lines = []
        self._render_node(tree.root, lines, "", is_last=True, is_root=True)
        return "\n".join(lines)
    
    def _render_node(
        self,
        node: FileNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
        is_root: bool = False
    ) -> None:
        if not is_root:
            connector = "└── " if is_last else "├── "
            icon = self._get_icon(node)
            line = f"{prefix}{connector}{icon} {node.name}"
            lines.append(line)
            
            extension = "    " if is_last else "│   "
            prefix = prefix + extension
        else:
            icon = self.dir_icon
            lines.append(f"{icon} {node.name}")
            prefix = ""
        
        if node.node_type == FileNodeType.DIRECTORY and node.is_expanded:
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._render_node(child, lines, prefix, is_last_child)
    
    def _get_icon(self, node: FileNode) -> str:
        if node.node_type == FileNodeType.DIRECTORY:
            if node.is_expanded:
                return f"{self.expanded_icon} {self.dir_icon}"
            else:
                return f"{self.collapsed_icon} {self.dir_icon}"
        elif node.is_neural_file:
            return self.neural_icon
        else:
            return self.file_icon
    
    def render_node_path(self, node: FileNode) -> str:
        parts = []
        current = node
        while current:
            parts.insert(0, current.name)
            current = current.parent
        return " / ".join(parts)


def example_tree_view():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        (temp_dir / "models").mkdir()
        (temp_dir / "models" / "cnn.neural").write_text("model CNN { }")
        (temp_dir / "models" / "rnn.neural").write_text("model RNN { }")
        
        (temp_dir / "data").mkdir()
        (temp_dir / "data" / "preprocessing.py").write_text("# preprocessing")
        
        (temp_dir / "config").mkdir()
        (temp_dir / "config" / "settings.yaml").write_text("backend: tensorflow")
        
        (temp_dir / "main.neural").write_text("model Main { }")
        (temp_dir / "README.md").write_text("# My Project")
        
        tree = FileTree(temp_dir)
        
        if tree.root:
            for child in tree.root.children:
                if child.node_type == FileNodeType.DIRECTORY:
                    tree.toggle_expand(child)
        
        renderer = TreeViewRenderer()
        output = renderer.render_tree(tree)
        
        print("File Tree View:")
        print(output)
        
        print("\n" + "="*50)
        print("Neural Files Only:")
        neural_files = tree.get_all_neural_files()
        for node in neural_files:
            path = renderer.render_node_path(node)
            print(f"  {renderer.neural_icon} {path}")
        
    finally:
        shutil.rmtree(temp_dir)


def example_interactive_tree():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "model1.neural").write_text("")
        (temp_dir / "src" / "model2.neural").write_text("")
        (temp_dir / "tests").mkdir()
        (temp_dir / "tests" / "test1.neural").write_text("")
        
        tree = FileTree(temp_dir)
        renderer = TreeViewRenderer()
        
        print("Initial Tree (collapsed):")
        print(renderer.render_tree(tree))
        
        if tree.root:
            for child in tree.root.children:
                if child.name == "src":
                    print(f"\nExpanding '{child.name}'...")
                    tree.toggle_expand(child)
                    tree.select_node(child)
        
        print("\nTree after expanding 'src':")
        print(renderer.render_tree(tree))
        
        if tree.selected_node:
            print(f"\nSelected node: {tree.selected_node.name}")
            print(f"Path: {renderer.render_node_path(tree.selected_node)}")
        
    finally:
        shutil.rmtree(temp_dir)


def example_tree_filtering():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / "__pycache__" / "cache.pyc").write_text("")
        
        (temp_dir / ".venv").mkdir()
        (temp_dir / ".venv" / "lib").mkdir()
        
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "model.neural").write_text("")
        (temp_dir / "src" / "script.py").write_text("")
        
        tree = FileTree(temp_dir)
        
        if tree.root:
            for child in tree.root.children:
                tree.toggle_expand(child)
        
        renderer = TreeViewRenderer()
        
        print("Filtered Tree (hidden: __pycache__, .venv):")
        print(renderer.render_tree(tree))
        print(f"\nTotal directories: {len([c for c in tree.root.children if c.node_type == FileNodeType.DIRECTORY])}")
        
    finally:
        shutil.rmtree(temp_dir)


def example_tree_operations():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "model.neural").write_text("")
        
        tree = FileTree(temp_dir)
        renderer = TreeViewRenderer()
        
        print("Initial Tree:")
        print(renderer.render_tree(tree))
        
        if tree.root:
            src_node = None
            for child in tree.root.children:
                if child.name == "src":
                    src_node = child
                    tree.toggle_expand(child)
                    break
            
            if src_node:
                print("\nAdding new file 'new_model.neural'...")
                new_node = tree.add_file(src_node, "new_model.neural")
                
                if new_node:
                    new_path = temp_dir / "src" / "new_model.neural"
                    new_path.write_text("model NewModel { }")
        
        print("\nTree after adding file:")
        print(renderer.render_tree(tree))
        
        if tree.root:
            print("\nAdding new directory 'tests'...")
            new_dir = tree.add_directory(tree.root, "tests")
            
            if new_dir:
                (temp_dir / "tests").mkdir()
                tree.toggle_expand(new_dir)
        
        print("\nTree after adding directory:")
        print(renderer.render_tree(tree))
        
    finally:
        shutil.rmtree(temp_dir)


def example_tree_with_callbacks():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        (temp_dir / "model1.neural").write_text("")
        (temp_dir / "model2.neural").write_text("")
        
        tree = FileTree(temp_dir)
        
        def on_selection_changed(node):
            if node:
                print(f"Selected: {node.name} ({node.node_type.value})")
            else:
                print("Selection cleared")
        
        tree.on_selection_changed = on_selection_changed
        
        if tree.root:
            for child in tree.root.children:
                tree.select_node(child)
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("="*60)
    print("Example 1: Basic Tree View")
    print("="*60)
    example_tree_view()
    
    print("\n" + "="*60)
    print("Example 2: Interactive Tree")
    print("="*60)
    example_interactive_tree()
    
    print("\n" + "="*60)
    print("Example 3: Tree Filtering")
    print("="*60)
    example_tree_filtering()
    
    print("\n" + "="*60)
    print("Example 4: Tree Operations")
    print("="*60)
    example_tree_operations()
    
    print("\n" + "="*60)
    print("Example 5: Tree with Callbacks")
    print("="*60)
    example_tree_with_callbacks()
