from setuptools import setup, find_packages
from setuptools_rust import RustExtension

setup(
    name="saif_dumper",
    version="0.1.0",
    description="SAIF (Switching Activity Interchange Format) dumper",
    author="Your Name",
    author_email="your.email@example.com",
    license="Apache 2.0",
    packages=find_packages(),
    rust_extensions=[RustExtension("saif_dumper")],
    install_requires=[],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
    ],
) 
 
