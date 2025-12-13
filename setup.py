# setup.py
from setuptools import find_packages, setup

# Core dependencies required for basic DSL functionality
CORE_DEPS = [
    "click>=8.1.3",
    "lark>=1.1.5",
    "numpy>=1.23.0",
    "pyyaml>=6.0.1",
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

CLOUD_DEPS = [
    "pygithub>=1.59",
    "selenium>=4.0",
    "webdriver-manager",
    "requests>=2.28.0",
]

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

UTILS_DEPS = [
    "psutil>=5.9.0",
    "pysnooper",
    "radon>=5.0",
    "pandas>=1.3",
    "scipy>=1.7",
    "statsmodels>=0.13",
    "sympy>=1.9",
    "multiprocess>=0.70",
]

ML_EXTRAS_DEPS = [
    "huggingface_hub>=0.16",
    "transformers>=4.30",
]

API_DEPS = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "flower>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.6",
    "pydantic-settings>=2.0.0",
    "requests>=2.31.0",
    "sqlalchemy>=2.0.0",
    "websockets>=10.0",
]

COLLABORATION_DEPS = [
    "websockets>=10.0",
]

MONITORING_DEPS = [
    "prometheus-client>=0.16.0",
    "requests>=2.28.0",
]

DATA_DEPS = [
    "dvc>=2.0",
    "pandas>=1.3",
]

TEAMS_DEPS = [
    "click>=8.1.3",
    "pyyaml>=6.0.1",
]

FEDERATED_DEPS = [
    "numpy>=1.23.0",
    "pyyaml>=6.0.1",
]

setup(
    name="neural-dsl",
    version="0.3.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=CORE_DEPS,
    extras_require={
        # Individual feature groups
        "hpo": HPO_DEPS,
        "automl": AUTOML_DEPS,
        "distributed": DISTRIBUTED_DEPS,
        "cloud": CLOUD_DEPS,
        "monitoring": MONITORING_DEPS,
        "integrations": INTEGRATION_DEPS,
        "visualization": VISUALIZATION_DEPS,
        "dashboard": DASHBOARD_DEPS,
        "backends": BACKEND_DEPS,
        "utils": UTILS_DEPS,
        "ml-extras": ML_EXTRAS_DEPS,
        "api": API_DEPS,
        "data": DATA_DEPS,
        "teams": TEAMS_DEPS,
        "federated": FEDERATED_DEPS,
        # Convenience bundles
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
            + CLOUD_DEPS
            + MONITORING_DEPS
            + INTEGRATION_DEPS
            + VISUALIZATION_DEPS
            + DASHBOARD_DEPS
            + BACKEND_DEPS
            + UTILS_DEPS
            + ML_EXTRAS_DEPS
            + API_DEPS
            + DATA_DEPS
            + TEAMS_DEPS
            + FEDERATED_DEPS
        ),
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.19",
            "myst-parser>=0.18"
        ]
    },
    entry_points={
        "console_scripts": ["neural=neural.__main__:cli"]
    },
    author="Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad",
    author_email="Lemniscate_zero@proton.me",
    description="A domain-specific language and debugger for neural networks",
    long_description=open("README.md", encoding="utf-8").read() + "\n\n**Note**: See v0.3.0 release notes for latest AI integration, deployment features, and automation improvements!",
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-world/Neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
