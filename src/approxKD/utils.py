""" THIS PYTHON FILE CONTAINS UTILITY FUNCTIONS """

# importing the required libraries
import numpy as np


# find the equidistant point from the two given vectors
def hyperplane_equation(v1:np.ndarray,v2:np.ndarray):
    """
    returns the equation of the hyperplane equidistant from v1 and v2

    PARAMETERS
    ----------
    v1 : vector1 (numpy ndarray)
    v2 : vector2 (numpy ndarray)

    RETURNS
    -------
    normal_vector  : normal vector to the hyperplane (numpy ndarray)
    constant       : constant term in the hyperplane equation (float)
    
    """
    # asserting  that v1 and v2 are numpy arrays
    assert isinstance(v1, np.ndarray),    "v1 must be a numpy array"
    assert isinstance(v2, np.ndarray),    "v2 must be a numpy array"
    assert v1.shape == v2.shape,          "v1 and v2 must have the same shape"
    
    # finding the normal vector
    normal_vector = v2-v1
    # finding the midpoint
    midpoint = (v1+v2)/2
    # finding the const term
    constant = np.dot(normal_vector,midpoint)
    return normal_vector,constant


# checking which side of the hyperplane does the vector lies
def check_vector_side(normal_vector,constant,vector):
    """
    Returns the side to the hyperplane does the vector lies

    PARAMETERS
    -----------
    normal_vector   : the vector normal to the hyperplane (numpy ndarray)
    constant        : constant term in the hyperplane (float)
    vector          : the vector whose side we want to find (numpy ndarray)

    RETURNS
    -------
    side            : side of the hyperplane does vector lies (string)
    
    """

    # Assert that normal_vector and vector are numpy arrays
    assert isinstance(normal_vector, np.ndarray),    "normal_vector must be a numpy array"
    assert isinstance(vector, np.ndarray),           "vector must be a numpy array"
    
    # performing the dot product between the normal vector and vector
    result = np.dot(normal_vector,vector)
    if result<constant:
        side="right"
    else:
        side="left"
    return side