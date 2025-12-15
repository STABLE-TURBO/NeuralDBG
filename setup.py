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

VISUALIZATION_DEPS = [
    "matplotlib>=3.5",
    "graphviz>=0.20",
    "networkx>=2.8",
    "plotly>=5.0",
]

DASHBOARD_DEPS = [
    "dash>=2.0",
    "flask>=2.0",
]

BACKEND_DEPS = [
    "torch>=1.10.0",
    "tensorflow>=2.6",
    "onnx>=1.10",
    "onnxruntime>=1.10",
]

AI_DEPS = [
    "langdetect>=1.0.9",
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
        "visualization": VISUALIZATION_DEPS,
        "dashboard": DASHBOARD_DEPS,
        "backends": BACKEND_DEPS,
        "ai": AI_DEPS,
        # Convenience bundles
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "full": (
            CORE_DEPS
            + HPO_DEPS
            + AUTOML_DEPS
            + DISTRIBUTED_DEPS
            + VISUALIZATION_DEPS
            + DASHBOARD_DEPS
            + BACKEND_DEPS
            + AI_DEPS
        ),
    },
    entry_points={
        "console_scripts": ["neural=neural.__main__:cli"]
    },
    author="Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad",
    author_email="Lemniscate_zero@proton.me",
    description="A domain-specific language and debugger for neural networks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-world/Neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
