# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg')
    img2 = misc.imread('image2.jpg')
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = img1.astype('float')
    img2 = img2.astype('float')
    
    def scale_range (input, min, max):
        output = np.copy(input)
        output += -(np.min(output))
        output /= np.max(output) / (max - min)
        output += min
        return output
    img1_scale = scale_range(img1,0,1)
    img2_scale = scale_range(img2,0,1)
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    img_comb = scale_range(img1_scale+img2_scale,0,1)
    misc.imsave('img_comb.jpg',img_comb)
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    h,w,_ = img1.shape
    newImage1 = np.hstack((img1_scale[:,:w/2,:],img2_scale[:,w/2:,:]))
    plt.imsave('newImage1.jpg',newImage1)
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros_like(img1)
    for i in range(h):
        if i % 2 == 0:
            newImage2[i,:] = img2_scale[i,:]
        else:
            newImage2[i,:] = img1_scale[i,:]
    plt.imsave('newImage2.jpg',newImage2)
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    img1_re = img1_scale.reshape(-1,w*2,3)
    img2_re = img2_scale.reshape(-1,w*2,3)
    newImage3 = np.stack((img2_re[:,:w,:],img1_re[:,w:,:]),axis=1).reshape(h,w,3)
    plt.imsave('newImage3.jpg',newImage3)
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.
    r, g, b = newImage3[:,:,0], newImage3[:,:,1], newImage3[:,:,2]
    gray = scale_range(0.2989 * r + 0.5870 * g + 0.1140 * b,0,255)
    plt.imshow(gray,cmap=plt.cm.Greys_r)
    plt.title('gray image3')
    plt.show()
    misc.imsave('gray_img3.jpg',gray)
    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()