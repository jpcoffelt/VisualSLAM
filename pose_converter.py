# Converts from original flattened transformation matrix
#
#       Rxx Rxy Rxz tx Ryx Ryy Ryz ty Rzx Rzy Rzz tz
#
# to simplified quaternion vector
#
#       tx ty tz qx qy qz qw

from scipy.spatial.transform import Rotation as R
import numpy as np


IN_FILE  = r"KITTI_sequence_2/poses.txt"
OUT_FILE = r"KITTI_sequence_2/poses_new.txt"


with open(IN_FILE, 'r') as in_f:
    with open(OUT_FILE, 'w') as out_f:
        for in_line in in_f.readlines():
            in_pose = np.asarray([float(val) for val in in_line.split()]).reshape((3, 4))
            R_mat = R.from_matrix(in_pose[:, :3])
            r_vec = np.asarray(R.as_quat(R_mat))
            t_vec = in_pose[:, -1]
            out_pose = np.concatenate((t_vec, r_vec))
            out_line = np.array2string(out_pose, formatter={'float_kind':lambda x: "%.6f" % x})[1:-1]
            print(out_line)
            out_f.write(out_line + "\n")
