# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    l , _ = real_XY.shape
    b = np.hstack((front_image,back_image)).reshape(-1,1)
    A = np.zeros((4*l,8))
    
    real_XY_ext = np.hstack((np.repeat(real_XY,2,axis=0),np.tile(np.array([[0],[150]]), (l,1))))
    real_XY_ext = np.hstack((real_XY_ext,np.ones((2*l,1))))
    for i in range(l):
        k = 4*i
        m = 2*i
        A[k,:4]   = real_XY_ext[m] 
        A[k+1,4:] = real_XY_ext[m]
        A[k+2,:4] = real_XY_ext[m+1]
        A[k+3,4:] = real_XY_ext[m+1]
    x = np.linalg.lstsq(A,b)
    camera_matrix = np.vstack((x[0].reshape(2,-1), np.array([0,0,0,1]) ))
    return camera_matrix

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    #TODO: Fill in this code
    l , _ = real_XY.shape
    real_XY_ext = np.hstack((np.repeat(real_XY,2,axis=0),np.tile(np.array([[0],[150]]), (l,1))))
    real_XY_ext = np.hstack((real_XY_ext,np.ones((2*l,1))))
    reprojective = np.dot(camera_matrix, real_XY_ext.T).T
    rms_error = reprojective[:,:2] - np.hstack((front_image,back_image)).reshape(-1,2)
    return np.sqrt(np.sum(np.square(rms_error)) / (2*l))
    
if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
