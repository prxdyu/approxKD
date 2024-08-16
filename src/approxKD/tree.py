
""" THIS PYTHON FILE CONTAINS THE NODE CLASS FOR THE KD-TREE """


# defining the class representing the Node of KD Tree
class Node:
    def __init__(self,hyperplane=None,constant=None,values=None):
        """
        PARAMETERS
        ----------
        hyperplane : the equation of the hyperplane equidistant from the two points
        constant   : the constant term in the hyperplane equation
        values     : the vectors to separate based on the hyperplane
        
        """
        self.hyperplane=hyperplane
        self.constant = constant
        self.values=values
        # vectors  that lie to the left hand side of the hyperplane
        self.left=None
        # vectors that lie to the right hand side of the hyperplane
        self.right=None
