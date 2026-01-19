"""
Setup script for granite-retrieval-agent with V2 orchestration
"""

from setuptools import setup, find_packages

setup(
    name="granite-retrieval-agent",
    version="2.0.0",
    description="AG2-based retrieval agent with V2 orchestration for Open WebUI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ag2==0.9.10",
        "aiohttp",
        "pydantic>=2.0",
    ],
    package_data={
        "v2": ["*.py", "**/*.py"],
    },
)
