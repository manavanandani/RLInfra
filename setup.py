from setuptools import setup, find_packages

setup(
    name="rlinfra",
    version="1.0.0",
    description="Distributed ML training platform with PyTorch DDP/FSDP and profiling",
    author="Manav Anandani",
    author_email="manavanandani304@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.22.0",
        "mlflow>=2.8.0",
        "PyYAML>=6.0.1",
        "rich>=13.7.0",
        "click>=8.1.7",
        "tabulate>=0.9.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "rlinfra=src.main:main",
        ],
    },
    python_requires=">=3.11",
)
