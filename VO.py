import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 

from lib.utils.pose_converter import cvt_pose_mat2vec, cvt_pose_vec2mat
from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir, use_quaternions=True):
        self.use_quaternions = use_quaternions
        if use_quaternions:
            poses_filename = "poses_vec.txt"
        else:
            poses_filename = "poses_mat.txt"

        self.K, self.P = self.load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self.load_poses(os.path.join(data_dir, poses_filename))
        self.images = self.load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def load_calib(self, filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def load_poses(self, filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                pose = np.fromstring(line, dtype=np.float64, sep=' ')
                if self.use_quaternions:
                    pose = cvt_pose_vec2mat(pose)
                else:
                    pose = pose.reshape((3, 4))
                    pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
        return poses

    def load_images(self, filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        

        #keypoints1 = self.orb.detect(self.images[i - 1], None)
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        #keypoints2 = self.orb.detect(self.images[i], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)



        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        # store all the good matches as per Lowe's ratio test.
        
        
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])

        return q1, q2

        # draw_params = dict(matchColor = -1, # draw matches in green color
        #         singlePointColor = None,
        #         matchesMask = None, # draw only inliers
        #         flags = 2)

        # img3 = cv2.drawMatches(self.images[i], keypoints1, self. images[i-1],keypoints2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(0)
        # plt.imshow(img3, 'gray'),plt.show()
        # plt.imshow(self.images[i]),plt.show()
        # plt.imshow(self.images[i-1]),plt.show()



        # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        pass

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """

        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        # print ("\nEssential matrix:\n" + str(Essential))

        R, t = self.decomp_essential_mat(Essential, q1, q2)

        return self._form_transf(R,t)

        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T
        pass

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)




def main():
    DATA_DIR = 'KITTI_sequence_2'  # Try KITTI_sequence_2 too
    USE_QUATERNIONS = True

    vo = VisualOdometry(DATA_DIR, use_quaternions=USE_QUATERNIONS)


    # play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    gt_poses = []
    est_poses = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            est_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            est_pose = np.matmul(est_pose, np.linalg.inv(transf))
            # print ("\nGround truth pose:\n" + str(gt_pose))
            # print ("\n Current pose:\n" + str(est_pose))
            # print ("The current pose used x,y: \n" + str(est_pose[0,3]) + "   " + str(est_pose[2,3]) )
        
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((est_pose[0, 3], est_pose[2, 3]))

        if USE_QUATERNIONS:
            _gt_pose = cvt_pose_mat2vec(gt_pose)
            _est_pose = cvt_pose_mat2vec(est_pose)          
        else:
            _gt_pose = gt_pose[:3, :].flatten()
            _est_pose = est_pose[:3, :].flatten()
        gt_poses.append(_gt_pose)
        est_poses.append(_est_pose)

    if USE_QUATERNIONS:
        prefix = "vec"
    else:
        prefix = "mat"


    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry",
                             file_out=os.path.join("output", os.path.basename(DATA_DIR), prefix + ".html"))

    np.savetxt(os.path.join("output", os.path.basename(DATA_DIR), prefix + "_gt.out"), gt_poses, delimiter=' ')
    np.savetxt(os.path.join("output", os.path.basename(DATA_DIR), prefix + "_est.out"), est_poses, delimiter=' ')
    np.save(os.path.join("output", os.path.basename(DATA_DIR), prefix + "_gt.npy"), gt_poses)
    np.save(os.path.join("output", os.path.basename(DATA_DIR), prefix + "_est.npy"), est_poses)

if __name__ == "__main__":
    main()
