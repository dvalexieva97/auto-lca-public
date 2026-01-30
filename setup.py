from setuptools import find_packages, setup

setup(
    name="auto-lca",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.13",
)
