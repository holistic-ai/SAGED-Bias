from setuptools import setup, find_packages

setup(
    name="saged-simplified",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "sqlalchemy>=1.4.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
        "saged>=0.1.0",  # Add the version of your SAGED package
    ],
    python_requires=">=3.8",
    author="SAGED Team",
    author_email="your.email@example.com",
    description="SAGED Simplified Application",
    long_description=open("README.md").read() if open("README.md").readable() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/saged-simplified",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 