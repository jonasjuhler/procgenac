from setuptools import find_packages, setup

install_requires = open("requirements.txt").read().strip().split("\n")

extras = {
    "test": [
        "flake8>=3.8.4",
        "black==20.8b1",
    ]
}

setup(
    name="procgenac",
    version="1.0",
    install_requires=install_requires,
    extras_require=extras,
    packages=find_packages(),
    python_requires=">=3.7",
    author_email="jonas.juhler.n@gmail.com",
    classifiers=["Programming Language :: Python :: 3.7"],
)
