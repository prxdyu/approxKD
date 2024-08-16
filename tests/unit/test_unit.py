import pytest
import numpy as np
from approxKD.utils import hyperplane_equation,check_vector_side
from approxKD.tree import Node
from approxKD.ann import KDTreeANN

# TEST CASE FOR UTILITY FUNCTIONS

test_hyperplane_inputs = [
                    (np.array([1, 2, 3]), np.array([4, 5, 6]), (3,), float),  # Test case 1: 3D vectors
                    (np.array([1, 0]), np.array([0, 1]), (2,), float),        # Test case 2: 2D orthogonal vectors
                    (np.array([1, 1, 1]), np.array([-1, -1, -1]), (3,), float), # Test case 3: Opposing vectors
                ]

test_vector_side_inputs = [
                    (np.array([1, 0]), 0, np.array([1, 1]), "left"),   # Vector is on the left side
                    (np.array([1, 0]), 0, np.array([-1, 1]), "right"), # Vector is on the right side
                    (np.array([0, 1]), 0, np.array([0, -1]), "right"), # Vector below the hyperplane
                    (np.array([0, 1]), 0, np.array([0, 1]), "left"),   # Vector above the hyperplane
                    (np.array([1, 1]), 1, np.array([0.5, 0.5]), "left"),  # Near the hyperplane but on the left
                    (np.array([1, 1]), 1, np.array([1.5, 1.5]), "left"),  # Clearly on the left side
                    (np.array([1, 1]), 1, np.array([-0.5, -0.5]), "right"), # Clearly on the right side
                ]


test_build_tree = [
                    (
                        [np.array([2, 3]), np.array([5, 4]), np.array([9, 6]), np.array([4, 7]), np.array([8, 1]), np.array([7, 2])],
                        1,
                        Node,
                        Node,
                        Node
                    ),
                    (
                        [np.array([1, 2]), np.array([3, 4])],
                        1,
                        Node,
                        Node,
                        Node
                    ),
                    (
                        [np.array([2, 3]), np.array([5, 4]), np.array([9, 6])],
                        2,
                        Node,
                        Node,
                        Node
                    ),
                ]


test_get_nearest_neighbors = [
                            (
                                [np.array([1, 0]), np.array([2, 1]), np.array([3, 2]), np.array([4, 3])],
                                np.array([3, 2]),
                                2,
                                "euclidean",
                                [np.array([3, 2]), np.array([2, 1])]
                            ),
                            (
                                [np.array([1, 0]), np.array([2, 1]), np.array([3, 2]), np.array([4, 3])],
                                np.array([3, 2]),
                                3,
                                "manhattan",
                                [np.array([3, 2]), np.array([2, 1]), np.array([4, 3])]
                            ),
                            (
                                [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])],
                                np.array([0, 0]),
                                2,
                                "cosine",
                                [np.array([1, 0]), np.array([0, 1])]
                            ),
                            (
                                [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])],
                                np.array([1, 1]),
                                1,
                                "euclidean",
                                [np.array([1, 1])]
                            ),
                            ]


@pytest.mark.parametrize("v1, v2, expected_normal_vector_shape, expected_constant_type",test_hyperplane_inputs)
def test_hyperplane_function(v1,v2,expected_normal_vector_shape, expected_constant_type):
    normal_vector,constant = hyperplane_equation(v1,v2)
    assert isinstance(normal_vector, np.ndarray)
    assert isinstance(constant,expected_constant_type)
    assert normal_vector.shape == expected_normal_vector_shape


@pytest.mark.parametrize("normal_vector, constant, vector, expected_side",test_vector_side_inputs)
def test_check_vector_side(normal_vector, constant, vector, expected_side):
    side = check_vector_side(normal_vector, constant, vector)
    assert side == expected_side



@pytest.mark.parametrize("vectors, min_subset_size, expected_node_type, expected_left_type, expected_right_type",test_build_tree)
def test_build_tree(vectors, min_subset_size, expected_node_type, expected_left_type, expected_right_type):
    kd_tree = KDTreeANN(min_subset_size=min_subset_size)
    root_node = kd_tree.build_tree(vectors)    
    assert isinstance(root_node, expected_node_type)
    if root_node.left:
        assert isinstance(root_node.left, expected_left_type)
    if root_node.right:
        assert isinstance(root_node.right, expected_right_type)
    assert root_node.values == vectors



@pytest.mark.parametrize("vectors, query_vector, k, metric, expected_neighbors",test_get_nearest_neighbors)
def test_get_nearest_neighbors(vectors, query_vector, k, metric, expected_neighbors):
    kd_tree = KDTreeANN(min_subset_size=1)
    neighbors = kd_tree.get_nearest_neighbors(vectors, query_vector, k, metric=metric)
    # Convert the numpy arrays to lists for easier comparison
    neighbors = [neighbor.tolist() for neighbor in neighbors]
    expected_neighbors = [neighbor.tolist() for neighbor in expected_neighbors]
    assert neighbors == expected_neighbors
    assert len(neighbors) == k