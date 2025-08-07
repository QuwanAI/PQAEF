# setup.py

from setuptools import setup, find_packages

setup(
    name="PQAEF",
    version="0.1",
    # author="PKU",
    # author_email="your.email@example.com",
    # description="A powerful tool for query-entity and evidence-finding tasks.",
    long_description=open('README.md').read(),
    # long_description_content_type="text/markdown",
    # url="https://github.com/your-username/DataCrafter",
    # license="MIT",
    
    # 告诉 setuptools 在哪里找到你的包
    # find_packages() 会自动在 'src' 目录下查找所有包
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.8",
    
    # 项目依赖项
    install_requires=[
        "numpy",
        "torch>=1.9",
        "transformers",
        # 在这里列出你的所有依赖
    ],
    
    # 命令行入口点
    entry_points={
        "console_scripts": [
            "pqaef-runner=PQAEF.run:main",
        ],
    },
    
    # 其他元数据
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)