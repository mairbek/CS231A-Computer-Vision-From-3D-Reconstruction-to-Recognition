# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    #TODO: Fill in this code
    
    # extract x, y
    x = points[:,0]
    y = points[:,1]
    
    # slope
    m1 = (y[1] - y[0]) / (x[1] - x[0]).astype(float)
    m2 = (y[3] - y[2]) / (x[3] - x[2]).astype(float)
    
    # lines
    l1 = np.array([m1,-1,y[0] - m1 * x[0]])
    l2 = np.array([m2,-1,y[2] - m2 * x[2]])
    
    # intersect point
    vp = np.cross(l1,l2)
    
    # euclidean coordinate
    vanishing_point = vp / vp[-1]
    return vanishing_point[:-1]

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    #TODO: Fill in this code
    A = np.ones((3,4))
    A[0,0]  = np.sum(vanishing_points[0] * vanishing_points[1])
    A[0,1:-1] = vanishing_points[0] + vanishing_points[1]
    A[1,0]  = np.sum(vanishing_points[0] * vanishing_points[-1])
    A[1,1:-1] = vanishing_points[0] + vanishing_points[-1]
    A[2,0]  = np.sum(vanishing_points[1] * vanishing_points[-1])
    A[2,1:-1] = vanishing_points[1] + vanishing_points[-1]
    
    u,s,vh = np.linalg.svd(A)
    w = vh[-1,:]
    W = np.array([[w[0],   0,   w[1]], 
                  [  0,  w[0],  w[2]], 
                  [w[1], w[2],  w[3]]])
    
    L = np.linalg.cholesky(W)
    K = np.linalg.inv(L.T)
    K = K / K[-1,-1]
    return K
'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    #TODO: Fill in this code
    
    # vanishing points inhomogeneous
    v1 = np.concatenate((vanishing_pair1[0],np.array([1])))
    v2 = np.concatenate((vanishing_pair1[1],np.array([1])))
    
    v3 = np.concatenate((vanishing_pair2[0],np.array([1])))
    v4 = np.concatenate((vanishing_pair2[1],np.array([1])))
    
    #vanishing lines
    l1 = np.cross(v1,v2).reshape(-1,1)
    l2 = np.cross(v3,v4).reshape(-1,1)
    
    w_star = np.dot(K ,K.T)
    
    # cosine
    den = np.sqrt(np.dot(np.dot(l1.T, w_star), l1)) * np.sqrt(np.dot(np.dot(l2.T, w_star), l2))
    nom = np.dot(np.dot(l1.T, w_star), l2)
    angle = np.arccos(nom/den)
    
    return np.degrees(angle)[0]

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    #TODO: Fill in this code
    def compute_d(v,K_inv):
        # v: (3,)
        # K: (3,3)
        # return: d (3,1)
        product = np.dot(K_inv, v)
        d = product / np.linalg.norm(product) 
        return d
    
    # orientation
    d1 = compute_d(np.concatenate((vanishing_points1[0],np.array([1]))),np.linalg.inv(K))
    d1_prime = compute_d(np.concatenate((vanishing_points2[0],np.array([1]))),np.linalg.inv(K))
    
    d2 = compute_d(np.concatenate((vanishing_points1[1],np.array([1]))),np.linalg.inv(K))
    d2_prime = compute_d(np.concatenate((vanishing_points2[1],np.array([1]))),np.linalg.inv(K))
    
    d3 = compute_d(np.concatenate((vanishing_points1[2],np.array([1]))),np.linalg.inv(K))
    d3_prime = compute_d(np.concatenate((vanishing_points2[2],np.array([1]))),np.linalg.inv(K))
    
    # linear equation
    A = np.zeros((9,9))
    A[0,:3]  = d1
    A[1,3:6] = d1
    A[2,6:]  = d1
    A[3,:3]  = d2
    A[4,3:6] = d2
    A[5,6:]  = d2
    A[6,:3]  = d3
    A[7,3:6] = d3
    A[8,6:]  = d3
    
    b = np.vstack((d1_prime, np.vstack((d2_prime ,d3_prime))))
    rotation_matrix = np.linalg.lstsq(A,b.reshape(-1,1))[0].reshape(3,-1)
    rotation_matrix = rotation_matrix / np.linalg.norm(rotation_matrix)
    return rotation_matrix
if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
