from setuptools import setup, find_packages

setup(
    name="carefree-ml",
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "carefree-data",
        "carefree-toolkit>=0.1.5",
        "dill", "future", "psutil", "pillow",
        "cython>=0.29.12", "numpy>=1.16.2", "scipy>=1.2.1",
        "scikit-learn>=0.20.3", "matplotlib>=3.0.3",
        "mkdocs", "mkdocs-material", "mkdocs-minify-plugin",
        "Pygments", "pymdown-extensions"
    ],
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    description="Machine Learning algorithms implemented with numpy",
    keywords="python machine-learning numpy"
)
