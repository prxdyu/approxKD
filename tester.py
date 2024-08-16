from src.approxKD.ann import KDTreeANN
import random
import numpy as np



# defining the minimum no of vectors allowed in a node
MIN_SUBSET_SIZE=5
DIMENSION = 384
N_VECTORS = 10000



# function to create a synthesized data
def generate_points(n,d):
    """ 
    generates n no of random vectors
    n : no of points to generate
    d : dimensionality of vectors

    """
    # defining the number of random embeddings (let us consider 1000 random embeddings)
    #n_vectors = 100
    #d = 10
    # initializing an empty list to store the embeddings
    vectors = []
    # generating 1000  random embeddings
    for i in range(n):
        # generate a random d-dim embeddings
        random_vector = [round(100 * random.random(), 2) for _ in range(d)]
        vectors.append(np.array(random_vector))
    return vectors



""" TESTING """
vectors = generate_points(N_VECTORS,DIMENSION)

obj = KDTreeANN(n_trees=3,min_subset_size=5)
trees = obj.build_kdtrees(vectors)


sample =  np.array([50 for _ in range(DIMENSION)])
nn__ = obj.get_approximate_neighbors(sample,trees)
print(f"The length of approximate neighbors are {len(nn__)}")

nn = obj.get_nearest_neighbors(nn__,sample,k=5)

print("The results are",nn)
