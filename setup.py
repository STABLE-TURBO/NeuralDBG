# setup.py
from setuptools import find_packages, setup


# Core dependencies required for basic DSL functionality
CORE_DEPS = [
    "click>=8.1.3",
    "lark>=1.1.5",
    "numpy>=1.23.0",
    "pyyaml>=6.0.1",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
]

# Optional dependencies grouped by feature
HPO_DEPS = [
    "optuna>=3.0",
    "scikit-learn>=1.0",
]

AUTOML_DEPS = [
    "optuna>=3.0",
    "scikit-learn>=1.0",
    "scipy>=1.7",
]

DISTRIBUTED_DEPS = [
    "ray>=2.0.0",
    "dask[distributed]>=2023.0.0",
]

# Cloud Platform Integrations (DEPRECATED - will be simplified)
CLOUD_DEPS = [
    "pygithub>=1.59",      # GitHub API
    "selenium>=4.0",       # Web automation
    "webdriver-manager",   # WebDriver setup
    "requests>=2.28.0",    # HTTP client
]

# ML Platform Integrations (major providers only)
INTEGRATION_DEPS = [
    "requests>=2.28.0",
    "boto3>=1.26.0",
    "google-cloud-aiplatform>=1.25.0",
    "google-cloud-storage>=2.10.0",
    "azure-ai-ml>=1.8.0",
    "azure-identity>=1.13.0",
]

VISUALIZATION_DEPS = [
    "matplotlib<3.10",
    "graphviz>=0.20",
    "networkx>=2.8.8",
    "plotly>=5.18",
    "seaborn>=0.11",
]

DASHBOARD_DEPS = [
    "dash>=2.18.2",
    "dash-bootstrap-components>=1.0.0",
    "flask>=3.0",
    "flask-cors>=3.1",
    "flask-httpauth>=4.4",
    "flask-socketio>=5.0.0",
    "python-dotenv>=1.0",
]

BACKEND_DEPS = [
    "torch>=1.10.0",
    "torchvision>=0.15",
    "tensorflow>=2.6",
    "onnx>=1.10",
    "onnxruntime>=1.10",
]

# Utility Libraries
UTILS_DEPS = [
    "psutil>=5.9.0",       # System monitoring
    "pysnooper",           # Debugging
    "radon>=5.0",          # Code metrics
    "pandas>=1.3",         # Data manipulation
    "scipy>=1.7",          # Scientific computing
    "statsmodels>=0.13",   # Statistics
    "sympy>=1.9",          # Symbolic math
    "multiprocess>=0.70",  # Multiprocessing
]

# HuggingFace Integration
ML_EXTRAS_DEPS = [
    "huggingface_hub>=0.16",   # Model hub
    "transformers>=4.30",      # Transformers
]

# REST API (Experimental)
API_DEPS = [
    "fastapi>=0.104.0",                    # API framework
    "uvicorn[standard]>=0.24.0",           # ASGI server
    "celery>=5.3.0",                       # Task queue
    "redis>=5.0.0",                        # Cache/queue
    "flower>=2.0.0",                       # Celery monitoring
    "python-jose[cryptography]>=3.3.0",    # JWT tokens
    "passlib[bcrypt]>=1.7.4",              # Password hashing
    "python-multipart>=0.0.6",             # File uploads
    "pydantic-settings>=2.0.0",            # Settings
    "requests>=2.31.0",                    # HTTP client
    "sqlalchemy>=2.0.0",                   # Database ORM
    "websockets>=10.0",                    # WebSocket support
]

# Collaboration (DEPRECATED)
COLLABORATION_DEPS = [
    "websockets>=10.0",
]

# Monitoring (Experimental)
MONITORING_DEPS = [
    "prometheus-client>=0.16.0",   # Metrics
    "requests>=2.28.0",            # HTTP client
]

# Data Versioning (Simplified)
DATA_DEPS = [
    "dvc>=2.0",        # Data version control
    "pandas>=1.3",     # Data manipulation
]

# Team Management
TEAMS_DEPS = [
    "click>=8.1.3",    # Already in core
    "pyyaml>=6.0.1",   # Already in core
]

# Federated Learning (DEPRECATED - will be extracted)
FEDERATED_DEPS = [
    "numpy>=1.23.0",   # Already in core
    "pyyaml>=6.0.1",   # Already in core
]

EDUCATION_DEPS = [
    "nbformat>=5.0",
    "jupyter>=1.0.0",
    "dash>=2.18.2",
    "dash-bootstrap-components>=1.0.0",
    "plotly>=5.18",
    "requests>=2.28.0",
]

setup(
    name="neural-dsl",
    version="0.3.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=CORE_DEPS,
    extras_require={
        # ==================================================================
        # Minimal Installation Tiers
        # ==================================================================
        "minimal": CORE_DEPS,  # Alias for clarity
        
        # Core features (recommended starting point)
        "core": (
            HPO_DEPS
            + VISUALIZATION_DEPS
            + DASHBOARD_DEPS
            + BACKEND_DEPS
        ),
        
        # ==================================================================
        # Individual Feature Groups
        # ==================================================================
        "hpo": HPO_DEPS,
        "automl": AUTOML_DEPS,
        "distributed": DISTRIBUTED_DEPS,
        "integrations": INTEGRATION_DEPS,
        "visualization": VISUALIZATION_DEPS,
        "dashboard": DASHBOARD_DEPS,
        "backends": BACKEND_DEPS,
        "utils": UTILS_DEPS,
        "ml-extras": ML_EXTRAS_DEPS,
        "data": DATA_DEPS,
        "teams": TEAMS_DEPS,
        "education": EDUCATION_DEPS,
        
        # ==================================================================
        # Experimental/Deprecated Features (use with caution)
        # ==================================================================
        "cloud": CLOUD_DEPS,              # Being simplified
        "monitoring": MONITORING_DEPS,    # Experimental
        "api": API_DEPS,                  # Experimental
        "collaboration": COLLABORATION_DEPS,  # DEPRECATED
        "federated": FEDERATED_DEPS,      # DEPRECATED (will be extracted)
        
        # ==================================================================
        # Convenience Bundles
        # ==================================================================
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "pylint>=2.15.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "pip-audit>=2.0.0",
        ],
        
        "full": (
            CORE_DEPS
            + HPO_DEPS
            + AUTOML_DEPS
            + DISTRIBUTED_DEPS
            + INTEGRATION_DEPS
            + VISUALIZATION_DEPS
            + DASHBOARD_DEPS
            + BACKEND_DEPS
            + UTILS_DEPS
            + ML_EXTRAS_DEPS
            + DATA_DEPS
            + TEAMS_DEPS
            + EDUCATION_DEPS
            # Note: Excluded experimental and deprecated features
        ),
        
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.19",
            "myst-parser>=0.18"
        ]
    },
    entry_points={
        "console_scripts": ["neural=neural.cli:cli"]
    },
    author="Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad",
    author_email="Lemniscate_zero@proton.me",
    description="A domain-specific language and debugger for neural networks",
    long_description=(
        open("README.md", encoding="utf-8").read()
        + "\n\n**Note**: See v0.3.0 release notes for latest AI integration, "
        "deployment features, and automation improvements!"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-world/Neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
