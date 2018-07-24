# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg','L')
    plt.imshow(img1,cmap=plt.cm.Greys_r)
    plt.show()
    u, s, vh = np.linalg.svd(img1, full_matrices=True)
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = np.dot((u[:,0] * s[0]).reshape(-1,1),vh[0].reshape(1,-1))
    plt.imshow(rank1approx,cmap=plt.cm.Greys_r)
    misc.imsave('rank1approx.jpg',rank1approx)
    plt.show()

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank20approx = np.dot(u[:,:20] * s[:20],vh[:20])
    plt.imshow(rank20approx,cmap=plt.cm.Greys_r)
    misc.imsave('rank20approx.jpg',rank20approx)
    plt.show()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()