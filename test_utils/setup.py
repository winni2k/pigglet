from setuptools import setup, find_packages

setup(
    name='pigglet_testing',
    version='0.1.0',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Scientists',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    description='Tools for testing pigglet',
    author='Warren Kretzschmar',
    author_email='warren.kretzschmar@ki.se',
    url='https://github.com/winni2k',
    install_requires=['networkx', 'numpy', 'pysam'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    python_requires='>=3.6, <4',

)
