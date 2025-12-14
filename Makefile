.PHONY: help clean clean-dry install test lint format

help:
	@echo "Neural DSL - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Repository Hygiene:"
	@echo "  make clean          - Remove 200+ redundant files (DESTRUCTIVE)"
	@echo "  make clean-dry      - Preview what would be removed (safe)"
	@echo ""
	@echo "Development:"
	@echo "  make install        - Install package with all dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linters (ruff)"
	@echo "  make format         - Format code with black/ruff"
	@echo ""
	@echo "Documentation:"
	@echo "  See REPOSITORY_HYGIENE.md for cleanup details"
	@echo "  See scripts/CLEANUP_README.md for script documentation"

clean:
	@echo "⚠️  WARNING: This will delete 200+ redundant files!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read dummy
	python scripts/cleanup_repository.py

clean-dry:
	@echo "Previewing cleanup (no files will be deleted)..."
	python scripts/cleanup_repository.py --dry-run || bash scripts/cleanup_repository.sh --dry-run

install:
	pip install -e ".[full]"

install-dev:
	pip install -e ".[full]"
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

lint:
	ruff check .
	pylint neural/

format:
	ruff check --fix .
	black neural/ tests/
