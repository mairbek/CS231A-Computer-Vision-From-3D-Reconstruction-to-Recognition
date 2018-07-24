# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ===== Problem 2a =====
    # Define Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M = np.vstack((np.arange(1,10).reshape(3,3), np.array([0, 2, 2])))
    a = np.array([[1],[1],[0]])
    b = np.array([[-1],[2],[5]])
    c = np.array([[0],[2],[3],[2]])
    # END YOUR CODE HERE

    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = np.dot(a.T,b)

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    # BEGIN YOUR CODE HERE
    a_ele_b = a * b
    # END YOUR CODE HERE

    # ===== Problem 2d =====
    # Find (a^T b)Ma
    aMb = aDotb *  np.matmul(M,a)
    # BEGIN YOUR CODE HERE
    
    # END YOUR CODE HERE

    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function repmat() may come in handy.

    newM = M * a.T

    # BEGIN YOUR CODE HERE
    
    # END YOUR CODE HERE

    # ===== Problem 2f =====
    # Without using a loop, sort all of the values
    # of M in increasing order and plot them.
    # Note we want you to use newM from e.

    # BEGIN YOUR CODE HERE
    plt.hist(np.sort(newM, axis=None))
    plt.show()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
