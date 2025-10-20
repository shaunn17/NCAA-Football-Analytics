"""
Setup script for NCAA Football Analytics Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="ncaa-football-analytics",
    version="0.1.0",
    author="NCAA Football Analytics Team",
    author_email="",
    description="An end-to-end data analytics platform for NCAA Division I college football",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ncaa-football-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "optuna>=3.2.0",
        ],
        "viz": [
            "plotly>=5.15.0",
            "dash>=2.14.0",
            "dash-bootstrap-components>=1.4.0",
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ncaa-collect=src.ingestion.data_collector:main",
            "ncaa-test-api=scripts.test_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
)


