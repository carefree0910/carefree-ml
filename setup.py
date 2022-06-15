from setuptools import setup
from setuptools import find_packages


VERSION = "0.1.2"

DESCRIPTION = "Machine Learning algorithms implemented with numpy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-ml",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "carefree-data>=0.2.7",
        "carefree-toolkit>=0.2.10",
        "pillow",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-ml",
    download_url=f"https://github.com/carefree0910/carefree-ml/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python machine-learning numpy",
)
