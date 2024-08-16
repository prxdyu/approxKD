# KDTreeANN

`KDTreeANN` is a Python package for approximate nearest neighbor search using KD-trees. It provides efficient methods to build KD-trees and find approximate nearest neighbors using various distance metrics.

## Installation

You can install `KDTreeANN` via pip from the PyPI repository:

```python
pip install KDTreeANN
```
## Usage
#### 1. Importing the package
```python
from approxKD.ann import KDTreeANN
```
#### 2. Initializing the Class
Create an instance of the KDTreeANN class:

```python
# Initialize with min_subset_size and number of trees
kdtree_ann = KDTreeANN(min_subset_size=10, n_trees=5)
```


#### 3. Building the KD-Trees
To build KD-trees from a list of vectors:

```python

# Build KD-trees
vectors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
trees = kdtree_ann.build_kdtrees(vectors)
```


#### 4. Finding Approximate Neighbors
To find approximate neighbors of a query vector:

```python
# Find approximate neighbors
query_vector = np.array([2, 3])
approx_neighbors = kdtree_ann.get_approximate_neighbors(query_vector, trees)
```


#### 5. Getting Approximate Nearest Neighbors
To find approximate nearest neighbors using KD-trees and KNN algorithm:

```python
# Get approximate nearest neighbors
approx_nearest_neighbors = kdtree_ann.get_approximate_nearest_neighbors(query_vector, trees, k=2, metric="euclidean")
```



