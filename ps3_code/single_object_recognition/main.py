import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from utils import *
import math
from collections import defaultdict

'''
MATCH_KEYPOINTS: Given two sets of descriptors corresponding to SIFT keypoints, 
find pairs of matching keypoints.

Note: Read Lowe's Keypoint matching, finding the closest keypoint is not
sufficient to find a match. thresh is the theshold for a valid match.

Arguments:
    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    threshold - The threshold which to accept from Lowe's Keypoint Matching
        algorithm

Returns:
    matches - An int ndarray of size (N, 2) of indices that for keypoints in 
        descriptors1 match which keypoints in descriptors2. For example, [7 5]
        would mean that the keypoint at index 7 of descriptors1 matches the
        keypoint at index 5 of descriptors2. Not every keypoint will necessarily
        have a match, so N is not the same as the number of rows in descriptors1
        or descriptors2. 
'''
def match_keypoints(descriptors1, descriptors2, threshold = 0.7):
    
    n = descriptors1.shape[0]
    matches = np.empty((0,2), int)
    for i in xrange(n):
        feature = descriptors1[i]
        distances = np.linalg.norm(descriptors2 - feature,axis=1)
        idx = np.argsort(distances)
        
        if distances[idx[0]] < threshold * distances[idx[1]]:
            match = np.array([i,idx[0]]).reshape(1,2)
            matches = np.vstack((matches,match))
    return matches        
        

'''
REFINE_MATCH: Filter out spurious matches between two images by using RANSAC
to find a projection matrix. 

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2. 

    reprojection_threshold - If the reprojection error is below this threshold,
        then we will count it as an inlier during the RANSAC process.

    num_iterations - The number of iterations we will run RANSAC for.

Returns:
    inliers - A vector of integer indices that correspond to the inliers of the
        final model found by RANSAC.

    model - The projection matrix H found by RANSAC that has the most number of
        inliers.
'''
def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
        num_iterations = 1000):
    
    n = matches.shape[0]
    seq = [i for i in range(n)]
    valid_sample = [i for i in range(4,n)]
    
    inliers = np.array([])
    H_best = np.zeros((3,3))
    
    pts_homo = np.ones((keypoints1.shape[0],3))
    pts_homo[:,:-1] = keypoints1[:,:2]
    for _ in xrange(num_iterations):
        
        # valid sample subset 
        sample_length = random.sample(valid_sample, 1)[0]
        sample = random.sample(seq,sample_length)
        
        candit = matches[sample]
        # 1st image coordinates
        pts = pts_homo[candit[:,0]]
        pts_prime = keypoints2[candit[:,1],:2]
        
        A = np.zeros((sample_length*2,9))
        
        # homography
        for i in xrange(sample_length):
            idx = 2 * i
            A[idx,3:6]  = -pts[i]
            A[idx,6:]   = pts[i] * pts_prime[i,1]
            A[idx+1,:3] = pts[i]
            A[idx+1,6:] = -pts[i] * pts_prime[i,0]
        
        u,s,vh = np.linalg.svd(A)
        H = vh[-1].reshape(3,3)
        
        # reprojection error
        pts_reproj_homo = H.dot(pts_homo[matches[:,0]].T)
        pts_reproj_inhomo = pts_reproj_homo[:-1,:] / pts_reproj_homo[-1,:]
        error = np.linalg.norm(keypoints2[matches[:,1],:2]-pts_reproj_inhomo.T,axis=1)
        inlier_idx = np.where(error < reprojection_threshold)[0]
        
        if len(inliers) < len(inlier_idx):
            inliers = inlier_idx
            H_best = H
        
    return inliers, H_best
'''
GET_OBJECT_REGION: Get the parameters for each of the predicted object
bounding box in the image

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2.

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

    thresh - The threshold we use in Hough voting to state that we have found
        a valid object region.

Returns:
    cx - A list of the x location of the center of the bounding boxes

    cy - A list of the y location of the center of the bounding boxes

    w - A list of the width of the bounding boxes

    h - A list of the height of the bounding boxes

    orient - A list of the orientation of the bounding box. Note that the 
        theta provided by the SIFT keypoint is inverted. You will need to
        re-invert it.
'''
def get_object_region(keypoints1, keypoints2, matches, obj_bbox, thresh = 5, 
        nbins = 4):
    #cx,cy,w,h,orient = [],[],[],[],[]
    kp1_match = keypoints1[matches[:,0]]
    kp2_match = keypoints2[matches[:,1]]
    
    # transfer into (x1,y1,w,h)
    bbox  = np.zeros(4)
    bbox[:2] = (obj_bbox[:2] + obj_bbox[2:])*0.5
    bbox[2:] = obj_bbox[2:] - obj_bbox[:2]
    
    # predicted bbox
    scale_ratio = kp2_match[:,2] / kp1_match[:,2]
    w2 = bbox[2] * scale_ratio
    h2 = bbox[3] * scale_ratio
    orient2 = kp2_match[:,-1] -  kp1_match[:,-1]
    x2 = kp2_match[:,0] + np.cos(orient2) * scale_ratio * (bbox[0] - 
                  kp1_match[:,0]) - np.sin(orient2) * scale_ratio * (bbox[1] - kp1_match[:,1])
    y2 = kp2_match[:,1] + np.sin(orient2) * scale_ratio * (bbox[0] - 
                  kp1_match[:,0]) + np.cos(orient2) * scale_ratio * (bbox[1] - kp1_match[:,1])
    
    #hough transform dim
    w2_max,w2_min = w2.max(),w2.min()
    #h2_max,h2_min = h2.max,h2.min
    o2_max,o2_min = orient2.max(),orient2.min()
    x2_max,x2_min = x2.max(),x2.min()
    y2_max,y2_min = y2.max(),y2.min()
    
    w2_binsize = 1.0 * (w2_max - w2_min) / nbins 
    o2_binsize = 1.0 * (o2_max - o2_min) / nbins
    x2_binsize = 1.0 * (x2_max - x2_min) / nbins
    y2_binsize = 1.0 * (y2_max - y2_min) / nbins
    
    bins = defaultdict(list)
    for t in range(matches.shape[0]):
        cx,cy,w,orient = x2[t],y2[t],w2[t],orient2[t]
        for i in range(nbins):
            for j in range(nbins):
                for m in range(nbins):
                    for n in range(nbins):
                        if(x2_min + i * x2_binsize <= cx <= x2_min+(i+1)*x2_binsize):
                            if(y2_min + j * y2_binsize <= cy <= y2_min+(j+1)*y2_binsize):
                                if(w2_min + m * w2_binsize <= w <= w2_min+(m+1)*w2_binsize):
                                    if(o2_min + n * o2_binsize <= orient <= o2_min+(n+1)*o2_binsize):
                                        bins[(i,j,m,n)].append(t)
    
    cx,cy,w,h,orient = [],[],[],[],[]
    for idx in bins:
        indices = bins[idx]
        votes = len(indices)
        
        if votes >= thresh:
            cx.append(np.sum(x2[indices]) / votes)
            cy.append(np.sum(y2[indices]) / votes)
            w.append(np.sum(w2[indices]) / votes)
            h.append(np.sum(h2[indices]) / votes)
            orient.append(np.sum(orient2[indices]) / votes)
    return cx,cy,w,h,orient

'''
MATCH_OBJECT: The pipeline for matching an object in one image with another

Arguments:
    im1 - The first image read in as a ndarray of size (H, W, C).

    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    im2 - The second image read in as a ndarray of size (H, W, C).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def match_object(im1, descriptors1, keypoints1, im2, descriptors2, keypoints2,
        obj_bbox):
    # Part A
    descriptors1, keypoints1, = select_keypoints_in_bbox(descriptors1,
        keypoints1, obj_bbox)
    matches = match_keypoints(descriptors1, descriptors2)
    plot_matches(im1, im2, keypoints1, keypoints2, matches)
    
    # Part B
    inliers, model = refine_match(keypoints1, keypoints2, matches)
    plot_matches(im1, im2, keypoints1, keypoints2, matches[inliers,:])

    # Part C
    cx, cy, w, h, orient = get_object_region(keypoints1, keypoints2,
        matches[inliers,:], obj_bbox)
    #plot_bbox([10,30], [20,50], [50,10], [100,30], [30,90], im2)
    plot_bbox(cx, cy, w, h, orient, im2)

if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('SIFT_data.mat')
    images = data['stopim'][0]
    obj_bbox = data['obj_bbox'][0]
    keypoints = data['keypt'][0]
    descriptors = data['sift_desc'][0]
    
    np.random.seed(0)

    for i in [2, 1, 3, 4]:
        match_object(images[0], descriptors[0], keypoints[0], images[i],
            descriptors[i], keypoints[i], obj_bbox)
