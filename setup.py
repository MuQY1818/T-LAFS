"""
Setup file for TLAFS package
"""

from setuptools import setup, find_packages

setup(
    name="tlafs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "torch>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "lightgbm>=3.1.0",
        "xgboost>=1.3.0",
        "catboost>=0.24.0",
        "pytorch-tabnet>=3.0.0",
        "google-generativeai>=0.1.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Time Series Learning and Feature Selection (TLAFS) Package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tlafs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 