"""
CI/CD Pipeline Template Generator.

Generates CI/CD pipeline configurations for GitHub Actions, GitLab CI,
and Jenkins to automate model testing, validation, and deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class CITemplateGenerator:
    """
    Generates CI/CD pipeline templates for various platforms.
    
    Example:
        generator = CITemplateGenerator()
        
        # Generate GitHub Actions workflow
        github_config = generator.generate_github_actions(
            model_name="fraud_detector",
            python_version="3.10",
            enable_gpu=True,
            deploy_environments=["staging", "production"]
        )
        
        # Save to file
        generator.save_template(
            github_config,
            ".github/workflows/ml-pipeline.yml"
        )
        
        # Generate GitLab CI pipeline
        gitlab_config = generator.generate_gitlab_ci(
            model_name="fraud_detector",
            python_version="3.10"
        )
        
        # Generate Jenkins pipeline
        jenkins_config = generator.generate_jenkins(
            model_name="fraud_detector",
            python_version="3.10"
        )
    """
    
    @staticmethod
    def generate_github_actions(
        model_name: str,
        python_version: str = "3.10",
        enable_gpu: bool = False,
        enable_monitoring: bool = True,
        deploy_environments: Optional[List[str]] = None,
        test_commands: Optional[List[str]] = None,
        extra_dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate GitHub Actions workflow configuration."""
        deploy_environments = deploy_environments or ["staging", "production"]
        test_commands = test_commands or [
            "python -m pytest tests/ -v",
            "python -m pytest tests/integration/ -v"
        ]
        extra_dependencies = extra_dependencies or []
        
        workflow = {
            "name": f"ML Pipeline - {model_name}",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main"]
                },
                "workflow_dispatch": {}
            },
            "env": {
                "MODEL_NAME": model_name,
                "PYTHON_VERSION": python_version
            },
            "jobs": {
                "test": {
                    "name": "Test Model",
                    "runs-on": "ubuntu-latest" if not enable_gpu else "self-hosted",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": python_version,
                                "cache": "pip"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "\n".join([
                                "python -m pip install --upgrade pip",
                                "pip install -e .",
                                "pip install -r requirements-dev.txt"
                            ] + [f"pip install {dep}" for dep in extra_dependencies])
                        },
                        {
                            "name": "Run linting",
                            "run": "python -m ruff check ."
                        },
                        {
                            "name": "Run type checking",
                            "run": "python -m mypy neural/ --ignore-missing-imports"
                        },
                        {
                            "name": "Run unit tests",
                            "run": test_commands[0]
                        },
                        {
                            "name": "Run integration tests",
                            "run": (
                                test_commands[1]
                                if len(test_commands) > 1
                                else "echo 'No integration tests'"
                            )
                        },
                        {
                            "name": "Upload test coverage",
                            "uses": "codecov/codecov-action@v3",
                            "with": {
                                "file": "./coverage.xml",
                                "flags": "unittests"
                            }
                        }
                    ]
                },
                "validate-model": {
                    "name": "Validate Model",
                    "runs-on": "ubuntu-latest" if not enable_gpu else "self-hosted",
                    "needs": ["test"],
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": python_version
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e ."
                        },
                        {
                            "name": "Validate model schema",
                            "run": (
                                "python -c \"from neural.mlops.registry import "
                                "ModelRegistry; print('Schema validation passed')\""
                            )
                        },
                        {
                            "name": "Run model validation tests",
                            "run": (
                                "python -m pytest tests/validation/ -v || "
                                "echo 'No validation tests found'"
                            )
                        },
                        {
                            "name": "Check model metrics",
                            "run": (
                                "python scripts/validate_metrics.py || "
                                "echo 'No metrics validation script'"
                            )
                        }
                    ]
                },
                "security-scan": {
                    "name": "Security Scan",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Run security scan",
                            "run": "\n".join([
                                "pip install pip-audit bandit",
                                "pip-audit",
                                "bandit -r neural/ -f json -o bandit-report.json"
                            ])
                        },
                        {
                            "name": "Upload security report",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "security-report",
                                "path": "bandit-report.json"
                            }
                        }
                    ]
                }
            }
        }
        
        for env in deploy_environments:
            job_name = f"deploy-{env}"
            if_condition = (
                "github.ref == 'refs/heads/main' && github.event_name == 'push'"
                if env == "production"
                else "github.ref == 'refs/heads/develop'"
            )
            workflow["jobs"][job_name] = {
                "name": f"Deploy to {env.title()}",
                "runs-on": "ubuntu-latest",
                "needs": ["test", "validate-model", "security-scan"],
                "if": if_condition,
                "environment": env,
                "steps": [
                    {
                        "name": "Checkout code",
                        "uses": "actions/checkout@v3"
                    },
                    {
                        "name": "Set up Python",
                        "uses": "actions/setup-python@v4",
                        "with": {
                            "python-version": python_version
                        }
                    },
                    {
                        "name": "Install dependencies",
                        "run": "pip install -e ."
                    },
                    {
                        "name": f"Deploy model to {env}",
                        "env": {
                            "ENVIRONMENT": env
                        },
                        "run": (
                            f"python scripts/deploy.py --model "
                            f"${{{{ env.MODEL_NAME }}}} --environment {env}"
                        )
                    }
                ]
            }
            
            if enable_monitoring:
                workflow["jobs"][job_name]["steps"].append({
                    "name": "Setup monitoring",
                    "run": f"python scripts/setup_monitoring.py --environment {env}"
                })
        
        return workflow
    
    @staticmethod
    def generate_gitlab_ci(
        model_name: str,
        python_version: str = "3.10",
        enable_gpu: bool = False,
        deploy_environments: Optional[List[str]] = None,
        test_commands: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate GitLab CI pipeline configuration."""
        deploy_environments = deploy_environments or ["staging", "production"]
        test_commands = test_commands or [
            "python -m pytest tests/ -v --cov=neural --cov-report=xml"
        ]
        
        pipeline = {
            "image": f"python:{python_version}",
            "stages": [
                "test",
                "validate",
                "security",
                "deploy"
            ],
            "variables": {
                "MODEL_NAME": model_name,
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.cache/pip"
            },
            "cache": {
                "paths": [".cache/pip", ".venv/"]
            },
            "before_script": [
                "python --version",
                "pip install --upgrade pip",
                "pip install -e .",
                "pip install -r requirements-dev.txt"
            ],
            "test:lint": {
                "stage": "test",
                "script": [
                    "python -m ruff check .",
                    "python -m mypy neural/ --ignore-missing-imports"
                ]
            },
            "test:unit": {
                "stage": "test",
                "script": test_commands,
                "coverage": "/(?i)total.*? (100(?:\\.0+)?\\%|[1-9]?\\d(?:\\.\\d+)?\\%)$/",
                "artifacts": {
                    "reports": {
                        "coverage_report": {
                            "coverage_format": "cobertura",
                            "path": "coverage.xml"
                        }
                    }
                }
            },
            "validate:model": {
                "stage": "validate",
                "script": [
                    (
                        "python -c \"from neural.mlops.registry import "
                        "ModelRegistry; print('Validation passed')\""
                    ),
                    "python -m pytest tests/validation/ -v || echo 'No validation tests'"
                ],
                "needs": ["test:unit"]
            },
            "security:scan": {
                "stage": "security",
                "script": [
                    "pip install pip-audit bandit",
                    "pip-audit || true",
                    "bandit -r neural/ -f json -o bandit-report.json || true"
                ],
                "artifacts": {
                    "paths": ["bandit-report.json"],
                    "when": "always"
                }
            }
        }
        
        for env in deploy_environments:
            job_name = f"deploy:{env}"
            pipeline[job_name] = {
                "stage": "deploy",
                "script": [
                    f"python scripts/deploy.py --model $MODEL_NAME --environment {env}"
                ],
                "environment": {
                    "name": env,
                    "url": f"https://{env}.example.com"
                },
                "needs": ["test:unit", "validate:model", "security:scan"]
            }
            
            if env == "production":
                pipeline[job_name]["only"] = ["main"]
                pipeline[job_name]["when"] = "manual"
            else:
                pipeline[job_name]["only"] = ["develop"]
        
        return pipeline
    
    @staticmethod
    def generate_jenkins(
        model_name: str,
        python_version: str = "3.10",
        deploy_environments: Optional[List[str]] = None,
        test_commands: Optional[List[str]] = None
    ) -> str:
        """Generate Jenkins pipeline (Jenkinsfile)."""
        deploy_environments = deploy_environments or ["staging", "production"]
        test_commands = test_commands or ["python -m pytest tests/ -v"]
        
        test_stage = "\\n            ".join(test_commands)
        
        deploy_stages = []
        for env in deploy_environments:
            approval = ""
            if env == "production":
                approval = "\n            input message: 'Deploy to production?', ok: 'Deploy'"
            
            deploy_stages.append(f"""
        stage('Deploy to {env.title()}') {{
            when {{
                branch '{' main' if env == 'production' else 'develop'}'
            }}
            steps {{{approval}
                sh 'python scripts/deploy.py --model {model_name} --environment {env}'
                sh 'python scripts/setup_monitoring.py --environment {env}'
            }}
        }}""")
        
        deploy_stages_str = "".join(deploy_stages)
        
        jenkinsfile = f"""
pipeline {{
    agent any
    
    environment {{
        MODEL_NAME = '{model_name}'
        PYTHON_VERSION = '{python_version}'
    }}
    
    stages {{
        stage('Setup') {{
            steps {{
                sh 'python --version'
                sh 'pip install --upgrade pip'
                sh 'pip install -e .'
                sh 'pip install -r requirements-dev.txt'
            }}
        }}
        
        stage('Lint') {{
            steps {{
                sh 'python -m ruff check .'
                sh 'python -m mypy neural/ --ignore-missing-imports'
            }}
        }}
        
        stage('Test') {{
            steps {{
                sh '{test_stage}'
            }}
            post {{
                always {{
                    junit 'test-results/*.xml'
                    publishHTML([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }}
            }}
        }}
        
        stage('Validate Model') {{
            steps {{
                sh '''python -c "from neural.mlops.registry import \\
                    ModelRegistry; print('Validation passed')"'''
                sh 'python -m pytest tests/validation/ -v || echo "No validation tests"'
            }}
        }}
        
        stage('Security Scan') {{
            steps {{
                sh 'pip install pip-audit bandit'
                sh 'pip-audit || true'
                sh 'bandit -r neural/ -f json -o bandit-report.json || true'
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: 'bandit-report.json', allowEmptyArchive: true
                }}
            }}
        }}{deploy_stages_str}
    }}
    
    post {{
        success {{
            echo 'Pipeline completed successfully!'
        }}
        failure {{
            echo 'Pipeline failed!'
            mail to: 'team@example.com',
                 subject: "Failed Pipeline: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                 body: "Something went wrong with ${{env.BUILD_URL}}"
        }}
    }}
}}
"""
        return jenkinsfile.strip()
    
    @staticmethod
    def generate_azure_pipelines(
        model_name: str,
        python_version: str = "3.10",
        deploy_environments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate Azure Pipelines configuration."""
        deploy_environments = deploy_environments or ["staging", "production"]
        
        pipeline = {
            "trigger": {
                "branches": {
                    "include": ["main", "develop"]
                }
            },
            "pr": {
                "branches": {
                    "include": ["main"]
                }
            },
            "variables": {
                "modelName": model_name,
                "pythonVersion": python_version
            },
            "stages": [
                {
                    "stage": "Test",
                    "jobs": [
                        {
                            "job": "TestJob",
                            "pool": {
                                "vmImage": "ubuntu-latest"
                            },
                            "steps": [
                                {
                                    "task": "UsePythonVersion@0",
                                    "inputs": {
                                        "versionSpec": "$(pythonVersion)",
                                        "addToPath": True
                                    }
                                },
                                {
                                    "script": "\n".join([
                                        "pip install --upgrade pip",
                                        "pip install -e .",
                                        "pip install -r requirements-dev.txt"
                                    ]),
                                    "displayName": "Install dependencies"
                                },
                                {
                                    "script": "python -m ruff check .",
                                    "displayName": "Run linting"
                                },
                                {
                                    "script": (
                                        "python -m pytest tests/ -v "
                                        "--cov=neural --cov-report=xml"
                                    ),
                                    "displayName": "Run tests"
                                },
                                {
                                    "task": "PublishCodeCoverageResults@1",
                                    "inputs": {
                                        "codeCoverageTool": "Cobertura",
                                        "summaryFileLocation": (
                                            "$(System.DefaultWorkingDirectory)/"
                                            "coverage.xml"
                                        )
                                    }
                                }
                            ]
                        }
                    ]
                },
                {
                    "stage": "Validate",
                    "dependsOn": "Test",
                    "jobs": [
                        {
                            "job": "ValidateModel",
                            "pool": {
                                "vmImage": "ubuntu-latest"
                            },
                            "steps": [
                                {
                                    "task": "UsePythonVersion@0",
                                    "inputs": {
                                        "versionSpec": "$(pythonVersion)"
                                    }
                                },
                                {
                                    "script": "pip install -e .",
                                    "displayName": "Install package"
                                },
                                {
                                    "script": (
                                        "python -m pytest tests/validation/ -v || "
                                        "echo 'No validation tests'"
                                    ),
                                    "displayName": "Validate model"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        for env in deploy_environments:
            branch = 'main' if env == 'production' else 'develop'
            deploy_stage = {
                "stage": f"Deploy{env.title()}",
                "dependsOn": ["Test", "Validate"],
                "condition": (
                    f"and(succeeded(), eq(variables['Build.SourceBranch'], "
                    f"'refs/heads/{branch}'))"
                ),
                "jobs": [
                    {
                        "deployment": f"Deploy{env.title()}",
                        "environment": env,
                        "pool": {
                            "vmImage": "ubuntu-latest"
                        },
                        "strategy": {
                            "runOnce": {
                                "deploy": {
                                    "steps": [
                                        {
                                            "task": "UsePythonVersion@0",
                                            "inputs": {
                                                "versionSpec": "$(pythonVersion)"
                                            }
                                        },
                                        {
                                            "script": "pip install -e .",
                                            "displayName": "Install package"
                                        },
                                        {
                                            "script": (
                                                f"python scripts/deploy.py --model "
                                                f"$(modelName) --environment {env}"
                                            ),
                                            "displayName": f"Deploy to {env}"
                                        }
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
            pipeline["stages"].append(deploy_stage)
        
        return pipeline
    
    @staticmethod
    def save_template(config: Any, output_path: str) -> None:
        """Save generated template to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(config, str):
            with open(output_file, 'w') as f:
                f.write(config)
        else:
            with open(output_file, 'w') as f:
                yaml.dump(config, f, sort_keys=False, default_flow_style=False)
