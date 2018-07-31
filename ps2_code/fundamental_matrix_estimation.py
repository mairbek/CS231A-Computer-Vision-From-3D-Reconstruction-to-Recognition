import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    r = points1.shape[0]
    A = np.zeros((r,9))

    for i in range(r):
        A[i] = np.dot(points2[i].reshape(3,-1),points1[i].reshape(1,-1)).flatten()
    u,s,vh = np.linalg.svd(A)
    # minimal error solution
    F_prime = vh[-1].reshape(3,3)
    # force fundamental matrix to rank 2
    u,s,vh = np.linalg.svd(F_prime)
    s[-1] = 0
    F = np.dot(u * s, vh)
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):

    def transform_matrix(pts):
        # T = [R 0;-RC 1]
        centriod = np.mean(pts,axis=0)
        diff = np.sum((pts - centriod)**2,axis=1)
        scaling = np.mean(np.sqrt(diff))
        R = np.diag([2/scaling,2/scaling])
        T = np.diag([2/scaling,2/scaling,1])
        T[:-1,-1] = - np.dot(R,centriod[:-1])
        return T

    T = transform_matrix(points1)
    T_prime = transform_matrix(points2)

    F_prime = lls_eight_point_alg(np.dot(T, points1.T).T,np.dot(T_prime, points2.T).T)
    F = np.dot(T_prime.T,np.dot(F_prime, T))
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image
    im2 - a HxW(xC) matrix that contains pixel values from the second image
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

#    # l = F * x_prime
#    # l_prime = F.T * x
#    l = np.dot(F,points2.T)
#    l_prime = np.dot(F.T, points1.T)
#    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#    ax1.imshow(im1,cmap='gray')
#    ax1.axis('off')
#    ax2.imshow(im2,cmap='gray')
#    ax1.axis('off')
    pSize=points1.shape
    n=pSize[0]
    #n=5
    #print 'points size :\n',pSize
    plt.figure()
    plt.imshow(im1, cmap ='gray')
    for i in range(n):
        plot_epipolar_line(im1,F,points2[i:i+1,:])       
        plt.plot(points1[i,0],points1[i,1],'o')
    #plt.axis('off')
    

    plt.figure()
    plt.imshow(im2, cmap ='gray')
    for i in range(n):
        #plot_epipolar_line(im2,F,points1)
        plot_epipolar_line(im2,F.T,points1[i:i+1,:])
        plt.plot(points2[i,0],points2[i,1],'o')


def plot_epipolar_line(img,F,points):
    m,n=img.shape[:2]
    line=F.dot(points.T)

    t=np.linspace(0,n,100)
    lt=np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    ndx=(lt>=0)&(lt<m)
    t=np.reshape(t,(100,1))
    #print 't\n',t[ndx]
    #print 'lt\n',lt[ndx]
    plt.plot(t[ndx],lt[ndx])
'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # abs(l.T * x) / sqrt(a**2 + b**2)
    l = np.dot(F.T,points2.T)
    coff = l[:-1]
    distances = np.abs(np.diag(np.dot(l.T,points1.T))) / np.sqrt(np.sum(coff**2,axis=0))
    average_distance = np.mean(distances)
    return average_distance

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
           compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
           compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
           F_normalized
        print "Distance to lines in image 1 for normalized:", \
           compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
           compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
