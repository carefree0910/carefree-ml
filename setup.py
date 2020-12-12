from setuptools import setup, find_packages

VERSION = "0.1.1"

DESCRIPTION = "Machine Learning algorithms implemented with numpy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-ml",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "carefree-data>=0.1.2",
        "carefree-toolkit>=0.1.5",
        "dill", "future", "psutil", "pillow",
        "cython>=0.29.12", "numpy>=1.16.2", "scipy>=1.2.1",
        "scikit-learn>=0.20.3", "matplotlib>=3.0.3",
    ],
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    url="https://github.com/carefree0910/carefree-ml",
    download_url=f"https://github.com/carefree0910/carefree-ml/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python machine-learning numpy"
)
