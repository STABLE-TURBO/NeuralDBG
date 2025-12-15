#!/bin/bash
# Shell script to remove cache directories and test artifacts
# Run this script to clean up development artifacts

echo "======================================================================"
echo "Cache and Artifacts Cleanup Script"
echo "======================================================================"
echo ""

# Function to remove directory with feedback
remove_dir() {
    local dir=$1
    local desc=$2
    if [ -d "$dir" ]; then
        echo "Removing $desc: $dir"
        rm -rf "$dir"
        if [ ! -d "$dir" ]; then
            echo "  ✓ Removed $desc"
            return 0
        else
            echo "  ✗ Failed to remove $desc"
            return 1
        fi
    fi
    return 2
}

# Function to remove files matching pattern
remove_files() {
    local pattern=$1
    local desc=$2
    local count=0
    
    echo "Searching for $desc..."
    while IFS= read -r -d '' file; do
        echo "Removing: $file"
        rm -f "$file"
        if [ ! -f "$file" ]; then
            ((count++))
        fi
    done < <(find . -name "$pattern" -type f -print0 2>/dev/null)
    
    echo "Removed $count $desc"
    return $count
}

echo "Step 1: Removing cache directories..."
echo "----------------------------------------------------------------------"
cache_total=0

# Remove __pycache__ directories
echo "Searching for __pycache__ directories..."
while IFS= read -r -d '' dir; do
    if remove_dir "$dir" "__pycache__"; then
        ((cache_total++))
    fi
done < <(find . -type d -name "__pycache__" -print0 2>/dev/null)

# Remove .pytest_cache directories
echo "Searching for .pytest_cache directories..."
while IFS= read -r -d '' dir; do
    if remove_dir "$dir" ".pytest_cache"; then
        ((cache_total++))
    fi
done < <(find . -type d -name ".pytest_cache" -print0 2>/dev/null)

# Remove .hypothesis directories
echo "Searching for .hypothesis directories..."
while IFS= read -r -d '' dir; do
    if remove_dir "$dir" ".hypothesis"; then
        ((cache_total++))
    fi
done < <(find . -type d -name ".hypothesis" -print0 2>/dev/null)

# Remove .mypy_cache directories
echo "Searching for .mypy_cache directories..."
while IFS= read -r -d '' dir; do
    if remove_dir "$dir" ".mypy_cache"; then
        ((cache_total++))
    fi
done < <(find . -type d -name ".mypy_cache" -print0 2>/dev/null)

# Remove .ruff_cache directories
echo "Searching for .ruff_cache directories..."
while IFS= read -r -d '' dir; do
    if remove_dir "$dir" ".ruff_cache"; then
        ((cache_total++))
    fi
done < <(find . -type d -name ".ruff_cache" -print0 2>/dev/null)

echo "Total cache directories removed: $cache_total"
echo ""

echo "Step 2: Removing virtual environment directories..."
echo "----------------------------------------------------------------------"
venv_total=0
for venv_dir in .venv .venv312 venv; do
    if remove_dir "$venv_dir" "virtual environment"; then
        ((venv_total++))
    fi
done
echo "Total virtual environments removed: $venv_total"
echo ""

echo "Step 3: Removing test artifacts..."
echo "----------------------------------------------------------------------"
artifact_total=0
remove_files "test_*.html" "test HTML files"
artifact_total=$?
remove_files "test_*.png" "test PNG files"
((artifact_total+=$?))
echo "Total test artifacts removed: $artifact_total"
echo ""

echo "Step 4: Removing temporary Python scripts..."
echo "----------------------------------------------------------------------"
script_total=0
for script in sample_tensorflow.py sample_pytorch.py; do
    if [ -f "$script" ]; then
        echo "Removing: $script"
        rm -f "$script"
        if [ ! -f "$script" ]; then
            echo "  ✓ Removed $script"
            ((script_total++))
        fi
    fi
done
echo "Total temporary scripts removed: $script_total"
echo ""

echo "======================================================================"
echo "Cleanup Complete!"
echo "======================================================================"
echo "Summary:"
echo "  - Cache directories: $cache_total"
echo "  - Virtual environments: $venv_total"
echo "  - Test artifacts: $artifact_total"
echo "  - Temporary scripts: $script_total"
echo ""
echo "All patterns are already in .gitignore and will not be tracked by git."
echo ""
