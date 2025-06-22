from setuptools import setup, find_packages

setup(
    name="model_wrappers",
    version="0.1.0",
    description="A custom model wrapper module",
    author="Qunzhong",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
)