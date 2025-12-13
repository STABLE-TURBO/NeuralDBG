from pathlib import Path
from .project_manager import ProjectManager
from .file_tree import FileTree
from .file_node import FileNode, FileNodeType
from .tab_manager import TabManager
from .file_operations import FileOperations
from .workspace_config import WorkspaceConfig
from .project_metadata import ProjectMetadata
from .recent_projects import RecentProjectsManager


def test_file_node_creation():
    path = Path("./test.neural")
    node = FileNode("test.neural", path, FileNodeType.FILE)
    assert node.name == "test.neural"
    assert node.path == path
    assert node.is_neural_file is True
    print("✓ FileNode creation test passed")


def test_tab_manager():
    tab_manager = TabManager()
    
    file1 = Path("./file1.neural")
    file2 = Path("./file2.neural")
    
    tab1 = tab_manager.open_tab(file1, "content1", activate=True)
    tab2 = tab_manager.open_tab(file2, "content2", activate=True)
    
    assert tab_manager.get_tab_count() == 2
    assert tab_manager.get_active_tab() == tab2
    
    tab_manager.activate_previous_tab()
    assert tab_manager.get_active_tab() == tab1
    
    tab_manager.update_tab_content(tab1, "modified content")
    assert tab1.is_modified is True
    
    modified_tabs = tab_manager.get_modified_tabs()
    assert len(modified_tabs) == 1
    
    print("✓ TabManager test passed")


def test_workspace_config():
    config = WorkspaceConfig(Path("./test_project"))
    
    assert config.get_compiler_backend() == "tensorflow"
    
    config.set_compiler_backend("pytorch")
    assert config.get_compiler_backend() == "pytorch"
    
    config.add_excluded_pattern("**/test")
    excluded = config.get_excluded_patterns()
    assert "**/test" in excluded
    
    print("✓ WorkspaceConfig test passed")


def test_project_metadata():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        metadata = ProjectMetadata(temp_dir)
        
        metadata.add_bookmark("test.neural", 10, "Test bookmark")
        bookmarks = metadata.get_bookmarks()
        assert len(bookmarks) == 1
        
        metadata.set_breakpoints("test.neural", [5, 10, 15])
        breakpoints = metadata.get_breakpoints("test.neural")
        assert "test.neural" in breakpoints
        assert len(breakpoints["test.neural"]) == 3
        
        metadata.add_recent_search("Conv2D")
        metadata.add_recent_search("Dense")
        searches = metadata.get_recent_searches()
        assert len(searches) == 2
        assert searches[0] == "Dense"
        
        metadata.set_setting("editor", "font_size", value=16)
        font_size = metadata.get_setting("editor", "font_size")
        assert font_size == 16
        
        assert metadata.save() is True
        assert (temp_dir / ".aquarium-project").exists()
        
        metadata2 = ProjectMetadata(temp_dir)
        assert metadata2.load() is True
        assert metadata2.get_setting("editor", "font_size") == 16
        
        print("✓ ProjectMetadata test passed")
    finally:
        shutil.rmtree(temp_dir)


def test_recent_projects():
    import tempfile
    import shutil
    
    temp_config_dir = Path(tempfile.mkdtemp())
    temp_project1 = Path(tempfile.mkdtemp())
    temp_project2 = Path(tempfile.mkdtemp())
    
    try:
        recent_mgr = RecentProjectsManager(temp_config_dir)
        
        recent_mgr.add_project(temp_project1, "Project 1")
        recent_mgr.add_project(temp_project2, "Project 2")
        
        projects = recent_mgr.get_all()
        assert len(projects) == 2
        
        assert recent_mgr.save() is True
        
        recent_mgr2 = RecentProjectsManager(temp_config_dir)
        assert recent_mgr2.load() is True
        projects2 = recent_mgr2.get_all()
        assert len(projects2) == 2
        
        print("✓ RecentProjectsManager test passed")
    finally:
        shutil.rmtree(temp_config_dir)
        shutil.rmtree(temp_project1)
        shutil.rmtree(temp_project2)


def test_file_operations():
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        file_ops = FileOperations()
        
        result = file_ops.new_file(temp_dir, "test.neural", "content")
        assert result.success is True
        assert result.path.exists()
        
        result2 = file_ops.read_file(result.path)
        assert result2[0] is True
        assert result2[1] == "content"
        
        result3 = file_ops.save_file(result.path, "new content")
        assert result3.success is True
        
        result4 = file_ops.read_file(result.path)
        assert result4[1] == "new content"
        
        new_name = "renamed.neural"
        result5 = file_ops.rename_file(result.path, new_name)
        assert result5.success is True
        assert (temp_dir / new_name).exists()
        
        print("✓ FileOperations test passed")
    finally:
        shutil.rmtree(temp_dir)


def test_project_manager_integration():
    import tempfile
    import shutil
    
    temp_config_dir = Path(tempfile.mkdtemp())
    temp_project = Path(tempfile.mkdtemp())
    
    try:
        pm = ProjectManager(temp_config_dir)
        
        assert pm.create_project(temp_project, "Test Project") is True
        assert pm.is_project_open() is True
        
        neural_files = pm.get_all_neural_files()
        assert len(neural_files) >= 1
        
        result = pm.new_file("test.neural")
        assert result.success is True
        
        if result.path:
            pm.open_file(result.path)
            assert pm.tab_manager.get_tab_count() >= 1
            
            active_tab = pm.tab_manager.get_active_tab()
            if active_tab:
                active_tab.content = "model Test { }"
                pm.save_file(active_tab)
        
        pm.close_project(save_state=True)
        assert pm.is_project_open() is False
        
        pm.open_project(temp_project)
        assert pm.is_project_open() is True
        
        print("✓ ProjectManager integration test passed")
    finally:
        shutil.rmtree(temp_config_dir)
        shutil.rmtree(temp_project)


def run_all_tests():
    print("Running project management system tests...\n")
    
    test_file_node_creation()
    test_tab_manager()
    test_workspace_config()
    test_project_metadata()
    test_recent_projects()
    test_file_operations()
    test_project_manager_integration()
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
