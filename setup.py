#!/usr/bin/env python3
"""
Setup script for infodynamics-jax.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="infodynamics-jax",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Inference as infodynamics - A general-purpose Bayesian inference library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/infodynamics-jax",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
        ],
    },
)
