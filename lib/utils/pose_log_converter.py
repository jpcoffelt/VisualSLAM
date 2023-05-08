# Converts from original flattened transformation matrix
#
#       Rxx Rxy Rxz tx Ryx Ryy Ryz ty Rzx Rzy Rzz tz
#
# to simplified quaternion vector
#
#       tx ty tz qx qy qz qw

from scipy.spatial.transform import Rotation as R
import numpy as np

from pose_converter import cvt_pose_mat2vec


IN_FILE  = r"../../KITTI_sequence_1/poses_mat.txt"
OUT_FILE = r"../../KITTI_sequence_1/poses_vec.txt"

IN_FILE  = r"../../KITTI_sequence_2/poses_mat.txt"
OUT_FILE = r"../../KITTI_sequence_2/poses_vec.txt"


with open(IN_FILE, 'r') as in_f:
    with open(OUT_FILE, 'w') as out_f:
        for in_line in in_f.readlines():
            in_pose = np.asarray([float(val) for val in in_line.split()])
            out_pose = cvt_pose_mat2vec(in_pose)
            out_line = np.array2string(out_pose, formatter={'float_kind':lambda x: "%.6f" % x})[1:-1]
            print(out_line)
            out_f.write(out_line + "\n")
