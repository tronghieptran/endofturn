import setuptools

setuptools.setup(
    name="endofturn",
    version="0.0.3",
    author="hieptrantrong",
    author_email="trantronghiep220597@gmail.com",
    description="Text Classification Package",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tronghieptran/endofturn",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas",
        "tensorflow >= 2.1.0",
        "keras >= 2.3.1",
        "torch >= 1.4.0",
        "transformers >= 2.8.0"
    ],
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)