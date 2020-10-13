from setuptools import find_packages, setup

setup(
    name="pigglet",
    description=(
        "The Phylogenetic Inference and genotyping from"
        " Genotype Likelihoods Tool"
    ),
    version="0.15.0",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Scientists",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    author="Warren Kretzschmar",
    author_email="warren.kretzschmar@ki.se",
    url="https://github.com/winni2k",
    install_requires=[
        "dataclasses",
        "networkx==2.4",
        "numpy",
        "numexpr",
        "pysam==0.16.0.1",
        "click==7.1.2",
        "h5py",
        "tqdm",
        "scipy==1.5.*",
        "msprime",
        "tskit",
        "pyfiglet==0.8.post1",
        "ray",
    ],
    packages=find_packages(
        "src", exclude=["contrib", "docs", "tests", "test_utils"]
    ),
    python_requires=">=3.8, <4",
    package_dir={"": "src"},
    entry_points="""
    [console_scripts]
    pigglet=pigglet.cli:cli
    """,
)
