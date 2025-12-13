from pathlib import Path
from .project_manager import ProjectManager
from .file_operations import FileOperationResult


def example_basic_usage():
    config_dir = Path.home() / ".aquarium"
    pm = ProjectManager(config_dir)
    
    project_path = Path("./my_neural_project")
    
    if pm.create_project(project_path, "My Neural Project"):
        print(f"Project created: {project_path}")
    
    neural_files = pm.get_all_neural_files()
    print(f"Found {len(neural_files)} .neural files")
    
    if neural_files:
        result = pm.open_file(neural_files[0])
        if result.success:
            print(f"Opened file: {neural_files[0].name}")
    
    active_tab = pm.tab_manager.get_active_tab()
    if active_tab:
        active_tab.content += "\n// Modified content"
        pm.tab_manager.update_tab_content(active_tab, active_tab.content)
    
    pm.save_file()
    
    pm.close_project()


def example_file_tree_navigation():
    pm = ProjectManager()
    project_path = Path("./my_neural_project")
    
    if pm.open_project(project_path):
        print("Project opened")
    
    if pm.file_tree and pm.file_tree.root:
        root = pm.file_tree.root
        print(f"Root: {root.name}")
        
        for child in root.children:
            print(f"  - {child.name} ({child.node_type.value})")
            
            if child.node_type.value == "directory":
                pm.file_tree.toggle_expand(child)


def example_multi_file_editing():
    pm = ProjectManager()
    project_path = Path("./my_neural_project")
    
    pm.open_project(project_path)
    
    file1 = project_path / "model1.neural"
    file2 = project_path / "model2.neural"
    file3 = project_path / "model3.neural"
    
    pm.open_file(file1)
    pm.open_file(file2)
    pm.open_file(file3)
    
    print(f"Open tabs: {pm.tab_manager.get_tab_count()}")
    
    pm.tab_manager.activate_next_tab()
    
    active = pm.tab_manager.get_active_tab()
    if active:
        print(f"Active tab: {active.title}")
    
    pm.save_file()
    
    pm.close_file()


def example_recent_projects():
    pm = ProjectManager()
    
    recent = pm.get_recent_projects(5)
    
    print("Recent Projects:")
    for project in recent:
        print(f"  - {project.name} ({project.path})")
        print(f"    Last opened: {project.last_opened}")


def example_workspace_configuration():
    pm = ProjectManager()
    project_path = Path("./my_neural_project")
    
    pm.open_project(project_path)
    
    if pm.workspace_config:
        backend = pm.workspace_config.get_compiler_backend()
        print(f"Compiler backend: {backend}")
        
        pm.workspace_config.set_compiler_backend("pytorch")
        
        pm.workspace_config.add_excluded_pattern("**/temp")
        
        excluded = pm.workspace_config.get_excluded_patterns()
        print(f"Excluded patterns: {excluded}")


def example_file_operations():
    pm = ProjectManager()
    project_path = Path("./my_neural_project")
    
    pm.open_project(project_path)
    
    result = pm.new_file("new_model.neural")
    if result.success:
        print(f"Created: {result.path}")
    
    if result.path:
        pm.open_file(result.path)
        
        active = pm.tab_manager.get_active_tab()
        if active:
            active.content = """model NewModel {
    input: [batch, 784]
    
    layer dense: Dense {
        units: 128
        activation: "relu"
    }
    
    layer output: Dense {
        units: 10
        activation: "softmax"
    }
}"""
            pm.save_file(active)
    
    old_path = project_path / "new_model.neural"
    rename_result = pm.rename_file(old_path, "renamed_model.neural")
    if rename_result.success:
        print(f"Renamed to: {rename_result.path}")


def example_project_metadata():
    pm = ProjectManager()
    project_path = Path("./my_neural_project")
    
    pm.open_project(project_path)
    
    if pm.project_metadata:
        pm.project_metadata.add_bookmark(
            "model.neural",
            42,
            "Important layer definition"
        )
        
        pm.project_metadata.set_breakpoints("model.neural", [10, 25, 42])
        
        pm.project_metadata.add_recent_search("Conv2D")
        
        pm.project_metadata.set_setting("editor", "font_size", value=16)
        
        pm.project_metadata.save()
        
        bookmarks = pm.project_metadata.get_bookmarks()
        print(f"Bookmarks: {len(bookmarks)}")


def example_callbacks():
    pm = ProjectManager()
    
    def on_project_opened(path):
        print(f"Project opened: {path}")
    
    def on_tab_opened(tab):
        print(f"Tab opened: {tab.title}")
    
    def on_tab_modified(tab):
        print(f"Tab modified: {tab.title}")
    
    pm.on_project_opened = on_project_opened
    pm.tab_manager.on_tab_opened = on_tab_opened
    pm.tab_manager.on_tab_modified = on_tab_modified
    
    project_path = Path("./my_neural_project")
    pm.open_project(project_path)
    
    neural_files = pm.get_all_neural_files()
    if neural_files:
        pm.open_file(neural_files[0])


if __name__ == "__main__":
    print("Running example_basic_usage...")
    example_basic_usage()
    
    print("\nRunning example_file_tree_navigation...")
    example_file_tree_navigation()
    
    print("\nRunning example_multi_file_editing...")
    example_multi_file_editing()
    
    print("\nRunning example_recent_projects...")
    example_recent_projects()
    
    print("\nRunning example_workspace_configuration...")
    example_workspace_configuration()
    
    print("\nRunning example_file_operations...")
    example_file_operations()
    
    print("\nRunning example_project_metadata...")
    example_project_metadata()
    
    print("\nRunning example_callbacks...")
    example_callbacks()
