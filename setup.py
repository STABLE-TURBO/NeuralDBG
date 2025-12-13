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

CLOUD_DEPS = [
    "pygithub>=1.59",
    "selenium>=4.0",
    "webdriver-manager",
    "tweepy>=4.15.0",
    "requests>=2.28.0",
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
    "fastapi>=0.68",
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
        "cloud": CLOUD_DEPS,
        "visualization": VISUALIZATION_DEPS,
        "dashboard": DASHBOARD_DEPS,
        "backends": BACKEND_DEPS,
        "utils": UTILS_DEPS,
        "ml-extras": ML_EXTRAS_DEPS,
        "api": API_DEPS,
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
            + CLOUD_DEPS
            + VISUALIZATION_DEPS
            + DASHBOARD_DEPS
            + BACKEND_DEPS
            + UTILS_DEPS
            + ML_EXTRAS_DEPS
            + API_DEPS
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
