import numpy as np
from scipy.spatial.transform import Rotation as R


def cvt_pose_mat2vec(mat):

# Converts a (possibly flattened/homogenized) pose matrix
#
#       Rxx Rxy Rxz tx
#       Ryx Ryy Ryz ty
#       Rzx Rzy Rzz tz
#      ----------------
#        0   0   0   1 <-- optional
#
# into a simplified quaternion vector
#
#       tx ty tz qx qy qz qw

    if mat.shape in [(12,), (12, 1), (16,), (16, 1)]: 

        mat = np.reshape(mat, (-1, 4))

    if mat.shape in [(3, 4), (4, 4)]: 
        R_mat = R.from_matrix(mat[:3, :3])
        t_vec = mat[:3, -1]
        r_vec = np.asarray(R.as_quat(R_mat))
        vec = np.concatenate((t_vec, r_vec))
        return vec
        
    else:
        print(mat.shape)
        print("[ERROR] Pose matrix has invalid dimensions!")
        return None
    

def cvt_pose_vec2mat(vec, homogenize=True, flatten=False):

# Converts a simplified quaternion pose vector
#
#       tx ty tz qx qy qz qw
#
# into a (possibly flattened/homogenized) matrix
#
#       Rxx Rxy Rxz tx
#       Ryx Ryy Ryz ty
#       Rzx Rzy Rzz tz
#      ----------------
#        0   0   0   1 <-- optional

    if vec.shape in [(7,), (7, 1)]: 

        t_vec = np.reshape(vec[:3], (3, 1))
        r_vec = R.from_quat(vec[3:])
        R_mat = np.asarray(R.as_matrix(r_vec))
        mat = np.concatenate((R_mat, t_vec), axis=1)

        if homogenize:
            mat = np.vstack((mat, [0, 0, 0, 1]))

        if flatten:
            mat = mat.flatten()

        return mat
        
    else:
        print("[ERROR] Pose vector has invalid dimensions!")
        return None
    

def test():

    vec = np.asarray([40.024030, -0.856448, 19.932530, 0.033282, 0.753438, 0.037567, 0.655601])

    mat0 = cvt_pose_vec2mat(vec, homogenize=False, flatten=False)
    mat1 = cvt_pose_vec2mat(vec, homogenize=False, flatten=True)
    mat2 = cvt_pose_vec2mat(vec, homogenize=True,  flatten=False)
    mat3 = cvt_pose_vec2mat(vec, homogenize=True,  flatten=True)

    print(mat0)
    print(mat1)
    print(mat2)
    print(mat3)

    print(cvt_pose_mat2vec(mat0))
    print(cvt_pose_mat2vec(mat1))
    print(cvt_pose_mat2vec(mat2))
    print(cvt_pose_mat2vec(mat3))


if __name__ == "__main__":
    test()


