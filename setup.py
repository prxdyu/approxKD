import setuptools

__version__="0.0.0"

REPO_NAME = "approxKD" 
AUTHOR_USER_NAME = "prxdyu"
AUTHOR_EMAIL = "pradyu1742@gmail.com"
SRC_REPO = "approxKD"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A python package that implements Approximate Nearest Neighbor algorithm using KD-Trees in python",
    long_description = "approxKD is a Python package for efficient Approximate Nearest Neighbor (ANN) searches using KD-trees. It provides tools to build KD-trees from multidimensional vectors and perform fast, approximate nearest neighbor searches. Key features include automatic hyperplane calculation, vector-side classification, and tree construction for quick spatial queries. Install via PyPI with pip install approxKD, and get started with simple examples provided in the documentation. For more details, visit project documentation",
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
                    "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
                    },
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src"),

)