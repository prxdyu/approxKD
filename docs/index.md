# approxKD 

This package is meant to find the approximate nearest neighbors using KD-Trees

- PYPI link for this package - [approxKD](https://pypi.org/project/approxKD/)

## Getting Started

### Installation

!!! note "installation steps"
    First let's do an easy pip installation of the library by running the following command -
    ```bash
    pip install approxKD
    ```


### Quickstart

!!! note "Quick start"
    First let's import the library -
    ```python
            # importing the library
            from approxKD.ann import KDTreeANN
    ```

    Now create an instance of KDTreeANN and build the KD-Trees using the build_kdtrees method
    ```python
            # creating an instance
            kdtree_ann = KDTreeANN(min_subset_size=10, n_trees=5)
            
            # building KD-trees
            vectors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
            trees = kdtree_ann.build_kdtrees(vectors)

    ```

    1. Finding the approximate neighbors using the get_approximate_neighbors method
    ```python
          
           # find approximate neighbors
           query_vector = np.array([2, 3])
           approx_neighbors = kdtree_ann.get_approximate_neighbors(query_vector, trees)
    ```

    2. Finding the approximate nearest neighbors  using the get_approximate_nearest_neighbors method
    ```python
           # getting the 2 nearest neighbors from approximate neighbors
           query_vector = np.array([2, 3])
           approx_nearest_neighbors = kdtree_ann.get_approximate_nearest_neighbors(query_vector, trees, k=2, metric="euclidean")
    ```



























