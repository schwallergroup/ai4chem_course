import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smiles_lstm",
    version="1.0.0",
    author="Rocío Mercado",
    author_email="rociomer@mit.edu",
    description="Molecular generative model based on a SMILES LSTM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rociomer/dl-chem-101/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)