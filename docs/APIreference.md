# API reference


# Defining the KDTreeANN instance
??? note "example"
    ### Short example
    ```python
    # importing the library
    from approxKD.ann import KDTreeANN
    
    # creating an instance of KDTreeANN
    kdtree_ann = KDTreeANN(min_subset_size=10, n_trees=5)
    ```



| Args               | Type | Description | 
|:------------------:|:----:|:------------|
| min_subset_size    | int  |minimum number of vectors to be in a node (if the number of vectors in a node is lesser than or equal to min_subset_size then the node will not be splitted further )|
| n_trees            | int  |number of KD-Trees to build |
 




## Building the KD-Tree 

??? note "example"
    ### Short example
    ```python
    # building kd-trees
    trees = kdtree_ann.build_kdtrees(vectors)
    ```

| Args    | Type  | Description | 
|:--------:|:------:|:-------|
| vectors | list  | list of the vectors |


| Returns   |Type  | Description | 
|:---------:|:----:|:-----|
| trees     | list | list of KD-Tree objects |



## Finding Approximate Neighbors

??? note "example"
    ### Short example
    ```python
    # finding the approximate neighbors
    approx_neighbors = kdtree_ann.get_approximate_neighbors(query_vector, trees)
    ```

| Args           | Type       | Description | 
|:--------------:|:----------:|:-------|
| query_vector   | np.ndarray |query vector whose neighbor we want to find |
| trees          | list       |list of KD-Tree objects |


| Returns   | Type   | Description | 
|:---------:|:------:|:------------|
| neighbors |  list  | List of approximate neighbors (np.ndarray) |




## Finding Approximate Nearest Neighbors

??? note "example"
    ### Short example
    ```python
        # get approximate nearest neighbors
        approx_nearest_neighbors = kdtree_ann.get_approximate_nearest_neighbors(query_vector, trees, k=2, metric="euclidean")
    ```

| Args           | Type       | Description | 
|:--------------:|:----------:|:-------|
| query_vector   | np.ndarray |query vector whose neighbor we want to find |
| trees          | list       |list of KD-Tree objects |
| k              | int        |number of nearest neighbors to find |
| metric         | str        |distance metric to use ("cosine","eucledian","manhattan") default is "cosine"|


| Returns   | Type   | Description | 
|:---------:|:------:|:------------|
| neighbors |  list  | List of (np.ndarrays) approximate nearest neighbors (np.ndarrays) |

