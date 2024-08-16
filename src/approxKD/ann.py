# importing the required libraries
import numpy as np
import random
from functools import reduce

from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean, cityblock

from approxKD.tree import Node
from approxKD.utils import hyperplane_equation,check_vector_side





# defining the class
class KDTreeANN:

    # constructor for initializing the object
    def __init__(self,min_subset_size,n_trees=1):
        assert n_trees>0,           " Minimum no of trees should be greater than or equal to 1 "
        assert min_subset_size>0,   " Minimum no of min subsets size should be greater than or equal to 1 "
        self.min_subset_size = min_subset_size
        self.n_trees = n_trees




    # building the tree
    def build_tree(self,vectors):
        """
        Recursive function builds the KD-tree using the given list of vectors

        PARAMETERS
        ----------
        vectors          : list of all vectors
        min_subset_size  : minimum number of points to be in a region
        
        
        RETURNS
        -------
        cuurent_node    : the root node of the KD Tree
        
        """

        # assertion statements 
        assert isinstance(vectors, list),          "vectors must be a list"
        assert len(vectors) > 0,                   "vectors list cannot be empty"

        for vector in vectors:
            assert isinstance(vector, np.ndarray), "Each element in vectors must be a numpy array"

        assert isinstance(self.min_subset_size, int), "min_subset_size must be an integer"
        assert self.min_subset_size >= 0, "min_subset_size must be non-negative"

        # initializing values for random indices
        idx1 = 0
        idx2 = 0
        while idx1==idx2:
            # pick any two random numbers withing the number of vectors range 
            idx1 = random.randint(0,len(vectors)-1)
            idx2 = random.randint(0,len(vectors)-1)
            
        # pick any two random vectors from the list of vectors 
        first_vector = vectors[idx1]
        second_vector = vectors[idx2]
        
        # find the equidistant hyperplane between first vector and second vector
        hyperplane,constant = hyperplane_equation(first_vector,second_vector)
        # classfiy all vectors in the vectors list are on left or right with respect to hyperplane
        left_nodes=[]
        right_nodes=[]
        for vector in vectors:
            side = check_vector_side(hyperplane,constant,vector)
            # appending the vectors that lie to the right side of the hyperplane in right nodes
            if side=="right":
                right_nodes.append(vector)
            elif side=="left":
                left_nodes.append(vector)
                
        # building the current  node
        current_node = Node(hyperplane=hyperplane,constant=constant,values=vectors)
        
        # if the size of left node is greater than the min_subset_size (if so we need to split it further)
        if len(left_nodes)>self.min_subset_size:
            current_node.left = self.build_tree(left_nodes)
        else:
            current_node.left = Node(values=left_nodes)

        # if the size of left node is greater than the min_subset_size (if so we need to split it further)
        if len(right_nodes)>self.min_subset_size:
            current_node.right = self.build_tree(right_nodes)
        else:
            current_node.right = Node(values=right_nodes)

        return current_node   
    


    
    
    # searching for the nearest neighbors through tree
    def traverse_tree(self,tree,query_vector):
        """
        computes the nearest neighbors of the query vector by traversing through the KD-tree

        PARAMETERS
        ----------
        tree         : ANNOY tree built using build_tree function (class <Node>)
        query_vector : vector whose nearest neighbors we want to find (numpy ndarray)

        RETURNS
        -------
        tree.values  : nearest neighbors of query vectors (list)
        

        leaf node has three condition 
        1. the size of values attribute of the leaf node is less than the min_subset_size 
        2. the hyperplane attribute of the leaf node is None (as there is no hyperplane needed to split the node further if the first conditon is satisfied )
        3. the constant attribute of the leaf node is None  (as there is no hyperplane needed to split the node further if the first conditon is satisfied )
        
        """

        # Assertion statements 
        assert isinstance(tree, Node), "tree must be an instance of the Node class"
        assert isinstance(query_vector, np.ndarray), "query_vector must be a numpy array"
        assert len(tree.values) > 0, "tree.values cannot be empty"
        assert tree.values[0].shape == query_vector.shape, "query_vector must have the same dimensions as the vectors in the tree"

        # traversing till we reach leaves
        while len(tree.values)>self.min_subset_size and tree.hyperplane is not None and tree.constant is not None:
            # checking the side of the vector sample
            side = check_vector_side(tree.hyperplane,tree.constant,query_vector)
            if side=="left":
                tree = tree.left
            elif side=="right":
                tree = tree.right
        
        return tree.values
    




    def build_kdtrees(self,vectors):
        """
        Function builds the N no of KD-trees using the given list of vectors

        PARAMETERS
        ----------
        vectors     : list of vectors (np.ndarray)
           
        RETURNS
        -------
        trees       : list of KD-Tree objects
        
        """
        # defining an empty list for storing KD Trees
        trees = []

        # building n_tree no of KD-Trees
        for _ in tqdm(range(self.n_trees), desc="Building KD-Trees", unit="tree"):
            tree = self.build_tree(vectors)
            trees.append(tree)
        
        return trees 



    
    def get_approximate_neighbors(self,query_vector,trees):
        """
        finds the approximate neighbors of the query vector by traversing through multiples KD-trees

        PARAMETERS
        ----------
        trees         : list of KD-Trees (class <list>)
        query_vector : vector whose approximate neighbors we want to find (numpy ndarray)

        RETURNS
        -------
        approx_neighbors  : approximate neighbors of query vectors (list)
        
        """
        
        assert isinstance(trees, list), "trees should be a list of KD-Tree objects."
        assert isinstance(query_vector, np.ndarray), "query_vector should be a numpy array."

        # defining an empty list to store the results 
        result = []
        for i in range(self.n_trees):
            # searching the approximate neighbors through KD-Tree
            nearest_vectors = self.traverse_tree(trees[i],query_vector)
            assert isinstance(nearest_vectors, list), "The output of traverse_tree should be a list of numpy arrays."

            tupled_vectors = [tuple(vector) for vector in nearest_vectors]
            result.append(tupled_vectors)
            
        # eliminating duplicate
        result_final = list(reduce(lambda x, y: x+y,result))
        approximate_neighbors = [np.array(t,dtype=np.float64) for t in set(result_final)] 
        return  approximate_neighbors
    

    


    def get_nearest_neighbors(self,vectors,query_vector,k,metric="cosine"):
        """
        finds the approximate nearest neighbors of the query vector by performing KNN algorithm on nearest neighbors

        PARAMETERS
        ----------
        vectors      : list of all vectors                                         (list)
        query_vector : vector whose approximate neighbors we want to find          (numpy.ndarray)
        metric       : distance metric to be used 'cosine','eucledian','manhattan' (str)
        k            : no of nearest neighbors to find                             (int)

        RETURNS
        -------
        nearest_neighbors  : nearest neighbors of query vectors (list)
        
        """
        assert isinstance(vectors, list),                               "vectors should be a list of numpy arrays."
        assert all(isinstance(vec, np.ndarray) for vec in vectors),     "Each element in vectors should be a numpy array."
        assert isinstance(query_vector, np.ndarray),                    "query_vector should be a numpy array."
        assert all(vec.shape == query_vector.shape for vec in vectors), "All vectors should have the same shape as query_vector."
        assert isinstance(k, int) and k > 0,                            "k should be a positive integer."
        assert k <= len(vectors),                                       "k cannot be greater than the number of available vectors."
        assert isinstance(metric, str),                                 "metric should be a string."
        
        # Dictionary to map metric names to functions
        metric_func = {
            "cosine": cosine,
            "euclidean": euclidean,
            "manhattan": cityblock
        }

        # Validate the metric
        if metric not in metric_func:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'cosine', 'euclidean', 'manhattan'.")


         # Validate the metric
        if metric not in metric_func:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'cosine', 'euclidean', 'manhattan'.")

        # Calculate distances from query_vector to all vectors
        distances = []
        for vector in vectors:
            distance = metric_func[metric](query_vector, vector)
            distances.append((distance, vector))

        # Sort the distances and get the top K nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [vector for _, vector in distances[:k]]

        return nearest_neighbors
    


    
    
    def get_approximate_nearest_neighbors(self,query_vector,trees,k,metric="cosine"):
        """
        finds the approximate nearest neighbors of the query vector by performing KNN algorithm on nearest neighbors

        PARAMETERS
        ----------

        query_vector                  : vector whose approximate neighbors we want to find          (numpy.ndarray)
        trees                         : list of KD-Trees (class <list>)
        metric                        : distance metric to be used 'cosine','eucledian','manhattan' (str)
        k                             : no of nearest neighbors to find                             (int)

        RETURNS
        -------
        approximate_nearest_neighbors : list of approximate nearest neighbors of query vectors (list)
        
        """
        assert isinstance(trees, list),                                 "trees should be a list."
        assert isinstance(query_vector, np.ndarray),                    "query_vector should be a numpy array."
        assert isinstance(k, int) and k > 0,                            "k should be a positive integer."
        assert isinstance(metric, str),                                 "metric should be a string."

        
        # getting the approximate nearest neighbors
        approximate_neighbors = self.get_approximate_neighbors(query_vector,trees)
        assert k <= len(approximate_neighbors),                         "k cannot be greater than the number of available neighbors."
        # getting the nearest neighbors from approximate neighbors
        approximate_nearest_neighbors = self.get_nearest_neighbors(approximate_neighbors,query_vector,k=2,metric=metric)
        return approximate_nearest_neighbors



