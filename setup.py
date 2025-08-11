# setup.py

from setuptools import setup, find_packages

setup(
    name="PQAEF",
    version="0.1",
    # author="PKU",
    long_description=open('README.md').read(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.8",
    
    # Project dependencies
    install_requires=[
        "numpy",
        "torch>=1.9",
        "transformers",
        # List all your dependencies here
    ],
    
    # Command line entry points
    entry_points={
        "console_scripts": [
            "pqaef-runner=PQAEF.run:main",
        ],
    },
    
    # Other metadata
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)