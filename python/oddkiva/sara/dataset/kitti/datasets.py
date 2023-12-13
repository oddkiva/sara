import os
import csv
from os import path
from pathlib import Path

import numpy as np
from scipy import linalg

import cv2


KITTI_ROOT_PATH = path.join(str(Path.home()), 'Downloads', 'kitti')
KITTI_LABELS_DIR_PATH = path.join(KITTI_ROOT_PATH, 'training', 'label_2')
KITTI_IMAGE_DIR_PATH = path.join(KITTI_ROOT_PATH, 'data_object_image_2',
                                 'training', 'image_2')
KITTI_CALIBRATION_DIR_PATH = path.join(KITTI_ROOT_PATH, 'data_object_calib',
                                       'training', 'calib')


def list_files(dir_path):
    filenames = os.listdir(dir_path)
    filepaths = [path.join(dir_path, f) for f in filenames]
    return filepaths


def read_labels(label_filepath):
    with open(label_filepath, newline='') as csv_file:
        labels = csv.reader(csv_file, delimiter=' ', quotechar="'")
        detections = []
        for row in labels:
            if row[0] == 'DontCare':
                continue
            print(row)
            (detection_type,
             # Truncation.
             truncation_score,
             # Occlusion state.
             occlusion_state,
             # Alpha.
             observation_angle,
             # 2D bounding box.
             left, top, right, bottom,
             # 3D oriented bounding box
             height, width, length,
             x, y, z,
             yaw_angle) = row
            detection = {
                'type': detection_type,
                'truncation_score': float(truncation_score),
                'occlusion_score': int(occlusion_state),
                'observation_angle_wrt_camera': float(observation_angle),
                'bbox_2d': ((float(left), float(top)),
                            (float(right), float(bottom))),
                'bbox_3d': ((float(x), float(y), float(z)),
                            (float(height), float(width), float(length)),
                            float(yaw_angle)),

                'score': float(1)
            }
            detections.append(detection)
    return detections


def read_calibration(calib_filepath):
    with open(calib_filepath, newline='') as csv_file:
        calibration_rows = csv.reader(csv_file, delimiter=' ', quotechar="'")
        calibration = {}
        for row in calibration_rows:
            if not row:
                continue

            name = row[0][:-1]
            data = [float(r) for r in row[1:]]
            calibration[name] = np.array(data)

            if name == 'R0_rect':
                calibration[name] = calibration[name].reshape((3, 3))
            else:
                calibration[name] = calibration[name].reshape((3, 4))

    P2 = calibration['P2']
    assert P2[0, 0] == P2[1, 1] and P2[2, 2] == 1

    print('P2 =\n', calibration['P2'])
    print('R0_rect =\n', calibration['R0_rect'])
    print('Tr_velo_to_cam =\n', calibration['Tr_velo_to_cam'])
    return calibration


def estimate_distance(pixel_coordinates, K_intrinsic_inv, C):
    p = np.dot(K_intrinsic_inv, pixel_coordinates)
    u, v = p[0], p[1]

    r11 = C[0, 0]
    r13 = C[0, 2]
    t1 = C[0, 3]
    r21 = C[1, 0]
    r23 = C[1, 2]
    t2 = C[1, 3]
    r31 = C[2, 0]
    r33 = C[2, 2]
    t3 = C[2, 3]

    A = np.array([[u * r31 - r11, u * r33 - r13],
                  [v * r31 - r21, v * r33 - r23]])
    b = np.array([-(u * t3 - t1),
                  -(v * t3 - t2)])
    LU = linalg.lu_factor(A)
    xz = linalg.lu_solve(LU, b)

    return xz


def bbox_local_coordinates_3d(l, w, h):
    """
    The coordinates of the 3D BBox in the local Euclidean object coordinate system.

    Consulting the MATLAB code is necessary to understand the meaning of
    each bounding box dimension.
    """
    Xs = [
        # Front face of the bbox
        #
        # Bottom-left.
        np.array([-l/2,  0, -w/2]),
        # Bottom-right.
        np.array([+l/2,  0, -w/2]),
        # Top-right.
        np.array([+l/2, -h, -w/2]),
        # Top-left.
        np.array([-l/2, -h, -w/2]),

        # Back face of the bbox.
        #
        # Bottom-left.
        np.array([-l/2,  0, +w/2]),
        # Bottom-right.
        np.array([+l/2,  0, +w/2]),
        # Top-right.
        np.array([+l/2, -h, +w/2]),
        # Top-left.
        np.array([-l/2, -h, +w/2]),
    ]
    return Xs


def bbox_camera_coordinates_3d(bbox_3d):
    (x, y, z), (h, w, l), yaw  = bbox_3d

    # Rotate the object w.r.t. the camera coordinate system.
    R = np.array([[+np.cos(yaw), 0, +np.sin(yaw)],
                  [           0, 1,            0],
                  [-np.sin(yaw), 0, +np.cos(yaw)]])
    # Translate the object w.r.t. the camera coordinate system.
    t = np.array([x, y, z])

    # The coordinates of the 3D BBox in the local Euclidean object coordinate system.
    #
    # Consulting the MATLAB code is necessary to understand the meaning of
    # each bounding box dimension.
    Xs = bbox_local_coordinates_3d(l, w, h)
    # print('3D local coordinates')
    # for X in Xs:
    #     print(X)

    # Rotate the object w.r.t. the camera coordinate system.
    R = np.array([[+np.cos(yaw), 0, +np.sin(yaw)],
                  [           0, 1,            0],
                  [-np.sin(yaw), 0, +np.cos(yaw)]])
    # Translate the object w.r.t. the camera coordinate system.
    t = np.array([x, y, z])

    # Transform to the camera coordinate system.
    Xs = [R @ X + t for X in Xs]
    print('Ground truth vertex coordinates in the camera frame')
    for X in Xs:
        print(X)

    return Xs


def draw_projected_bbox(image, xs):
    # Draw the corner of the bbox.
    for x in xs:
        cv2.circle(image, (int(x[0]), int(x[1])), 3, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)

    # Draw the edges of the front face.
    for i in range(4):
        xa, ya = xs[i][:2].astype(np.int)
        xb, yb = xs[(i + 1) % 4][:2].astype(np.int)
        cv2.line(image, (xa, ya), (xb, yb), (0, 0, 255), 1, cv2.LINE_AA)

    # Draw the edges of the back face.
    for i in range(4):
        xa, ya = xs[4 + i][:2].astype(np.int)
        xb, yb = xs[4 + (i + 1) % 4][:2].astype(np.int)
        cv2.line(image, (xa, ya), (xb, yb), (0, 0, 127), 1, cv2.LINE_AA)

    # Draw the edges of the left and right side faces.
    for i in range(4):
        (xa, ya) = xs[0 + i][:2].astype(np.int)
        (xb, yb) = xs[4 + i][:2].astype(np.int)
        cv2.line(image, (xa, ya), (xb, yb), (0, 0, 191), 1, cv2.LINE_AA)


image_filepaths = list_files(KITTI_IMAGE_DIR_PATH)
data_names = [path.splitext(path.basename(f))[0] for f in image_filepaths]
data_names.sort()

image_filepaths = [path.join(KITTI_IMAGE_DIR_PATH, '{}.png'.format(data_name))
                   for data_name in data_names]
label_filepaths = [path.join(KITTI_LABELS_DIR_PATH, '{}.txt'.format(data_name))
                   for data_name in data_names]
calibration_filepaths = [path.join(KITTI_CALIBRATION_DIR_PATH, '{}.txt'.format(data_name))
                         for data_name in data_names]


for (label_filepath, calib_filepath, image_filepath) in zip(label_filepaths,
                                                            calibration_filepaths,
                                                            image_filepaths):
    detections = read_labels(label_filepath)
    calibration = read_calibration(calib_filepath)
    image = cv2.imread(image_filepath)

    P2 = calibration['P2']

    R0_rect = calibration['R0_rect']
    R0_rect_4 = np.zeros((4, 4))
    R0_rect_4[:3, :3] = R0_rect
    R0_rect_4[3, 3] = 1

    # The actual projection matrix of 3D vertices
    # P = P2 @ R0_rect_4
    #
    # Decompose this projection matrix as K [R, t].
    K = P2[:3, :3]
    R = R0_rect
    t = np.linalg.inv(K) @ P2[:, 3]
    # Cameras are 1m65 above the ground.
    t[1] += 1.65
    print('Calibration matrix: K =\n', K)
    print('R =\n', R)
    print('t = ', t)
    K_inverse = np.linalg.inv(K)

    # The camera matrix (without the calibration matrix) is said above:
    camera_matrix = np.zeros((3, 4))
    camera_matrix[:, :3] = R
    camera_matrix[:, 3] = t
    print('Camera matrix: C = \n', camera_matrix)


    for detection in detections:
        if detection['type'] == 'DontCare':
            continue

        # Draw the 2D bbox.
        bbox_2d = detection['bbox_2d']
        (x1, y1), (x2, y2) = bbox_2d
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 0, 0), 2,
                      cv2.LINE_AA)

        # Bottom line of the bbbox
        ground_point = np.array((0.5 * (x1 + x2), y2, 1))
        xz = estimate_distance(ground_point, K_inverse, camera_matrix)

        # Populate the vertices of the 3D bbox in the camera frame.
        bbox_3d = detection['bbox_3d']
        Xs = bbox_camera_coordinates_3d(bbox_3d)

        print("Estimated xz = ", xz)
        print("")

        # Project the corners of the 3D bbox to the image.
        Xs = [np.concatenate((X, np.ones(1))) for X in Xs]
        xs = [P2 @ R0_rect_4 @ X for X in Xs]
        xs = [x / x[2] for x in xs]

        # Draw the center of the BBox.
        C = np.average(Xs, 0)
        c = P2 @ R0_rect_4 @ C
        c /= c[2]
        cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)

        # Draw the projected 3D bbox on the image.
        draw_projected_bbox(image, xs)

        cv2.putText(image, "x={:.2f}m z={:.2f}m".format(xz[0], xz[1]),
                    (int(c[0]), int(c[1])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(image, "x={:.2f}m z={:.2f}m".format(xz[0], xz[1]),
                    (int(c[0]), int(c[1])),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key == 27:
        break
