#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import itertools
import math
import random
from typing import List, Optional, Tuple

import numpy as np
import cv2
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    compute_reprojection_errors, eye3x4
)


def triangulate(intrinsic_mat, views, points_2d):
    arr = np.empty(shape=(2 * len(points_2d), 4), dtype=float)

    for i, (view, point_2d) in enumerate(zip(views, points_2d)):
        p = intrinsic_mat @ view
        arr[i * 2] = p[2] * point_2d[0] - p[0]
        arr[i * 2 + 1] = p[2] * point_2d[1] - p[1]

    res = np.linalg.lstsq(arr[:, :3], -arr[:, 3], rcond=None)
    return res[0]


def refine_points_3d(known_points3d, known_corners, known_views, known_frames, corner_storage, intrinsic_mat):
    threshold = 5

    for idx, corner_id in enumerate(known_corners):
        points_2d = []
        views = []

        for frame in known_frames:
            if corner_id in corner_storage[frame].ids.flatten():
                views.append(known_views[frame])
                point_2d = corner_storage[frame].points[(corner_storage[frame].ids.flatten() == corner_id)][0]
                points_2d.append(point_2d)

        if len(points_2d) >= threshold:
            new_point_3d = triangulate(intrinsic_mat, views, points_2d)
            known_points3d[idx] = new_point_3d


def refine_views(known_frames, corner_storage, known_corners, known_points3d, intrinsic_mat, known_views, inliers):
    inliers_list = list(inliers)
    for frame in known_frames:
        flag = np.isin(corner_storage[frame].ids.flatten(), inliers_list)

        flag2 = np.isin(known_corners, corner_storage[frame].ids.flatten()[flag])

        if flag2.sum() < 4:
            continue

        new_view = cv2.solvePnPRansac(known_points3d[flag2], corner_storage[frame].points[flag],
                                      intrinsic_mat, None, reprojectionError=8.0, confidence=0.99,
                                      iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)

        if not new_view[0]:
            continue

        view_mat = rodrigues_and_translation_to_view_mat3x4(new_view[1], new_view[2])
        known_views[frame] = view_mat


def triangulate_2(frames, corner_storage, known_corners, known_views, intrinsic_mat, min_angle, last_frame):
    known_corners = np.sort(known_corners)
    max_new_points = 0
    result_points3d = None
    result_ids = None
    res_angle = None

    for frame in frames:
        if frame == last_frame:
            continue

        frame1 = frame
        frame2 = last_frame

        correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2], known_corners)

        if len(correspondences.ids) < 1:
            continue

        points3d, ids, median_cos = triangulate_correspondences(
            correspondences,
            known_views[frame1],
            known_views[frame2],
            intrinsic_mat,
            TriangulationParameters(8.0, 0.5, 0.2)
        )

        angle = math.acos(median_cos) * 180 / math.pi

        if angle < min_angle:
            continue

        if len(points3d) > max_new_points:
            max_new_points = len(points3d)
            result_points3d = points3d
            result_ids = ids
            res_angle = angle

    result = result_points3d is not None
    return result, result_points3d, result_ids, res_angle


def pnp(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, inliers):
    max_inliers = -1
    max_corners = -1
    result_frame_inliers = None
    result_frame = None

    for frame in frames_queue:
        flag = np.isin(corner_storage[frame].ids.flatten(), inliers)

        if flag.sum() > max_inliers:
            max_inliers = flag.sum()
            result_frame_inliers = frame

        flag = np.isin(corner_storage[frame].ids.flatten(), known_corners)

        if flag.sum() > max_corners:
            max_corners = flag.sum()
            result_frame = frame

    if max_inliers >= 6:
        frame = result_frame_inliers
        corner_list = inliers
    else:
        frame = result_frame
        corner_list = known_corners

    print(len(corner_list), max_inliers)
    flag = np.isin(corner_storage[frame].ids.flatten(), corner_list)
    flag2 = np.isin(known_corners, corner_storage[frame].ids.flatten()[flag])

    new_view = cv2.solvePnPRansac(known_points3d[flag2], corner_storage[frame].points[flag],
                            intrinsic_mat, None, flags=cv2.SOLVEPNP_ITERATIVE)
    view_mat = rodrigues_and_translation_to_view_mat3x4(new_view[1], new_view[2])

    print(f"New frame №{frame}. Inliers: {len(inliers)}")

    return frame, view_mat


def pnp_ransac(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, min_corners, inliers):
    frames = list(frames_queue)
    np.random.shuffle(frames)

    known_points3d_cp = known_points3d.copy()
    known_corners_cp = known_corners.copy()

    for frame in frames_queue:
        flag = np.isin(corner_storage[frame].ids.flatten(), known_corners)

        if flag.sum() < min_corners:
            continue

        flag2 = np.isin(known_corners, corner_storage[frame].ids.flatten()[flag])
        if flag2.sum() < 4:
            continue
        new_view = cv2.solvePnPRansac(known_points3d[flag2], corner_storage[frame].points[flag],
                                      intrinsic_mat, None, reprojectionError=8.0, confidence=0.99,
                                      iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)

        if not new_view[0]:
            continue

        view_mat = rodrigues_and_translation_to_view_mat3x4(new_view[1], new_view[2])

        error = compute_reprojection_errors(known_points3d[flag2], corner_storage[frame].points[flag],
                                            intrinsic_mat @ view_mat).mean()

        outliers = np.delete(np.arange(flag2.sum()), new_view[3].flatten())
        outliers_ids = corner_storage[frame].ids.flatten()[flag][outliers]
        outlier_flag = np.isin(known_corners, outliers_ids)
        known_corners = known_corners[np.logical_not(outlier_flag)]
        known_points3d = known_points3d[np.logical_not(outlier_flag)]

        if error > 10.0:
            continue

        print(f"New frame №{frame}. Inliers: {len(new_view[3])}")

        inliers.update(corner_storage[frame].ids[flag][new_view[3].flatten()].flatten())
        return frame, view_mat

    return pnp(frames_queue, corner_storage, known_corners_cp, known_points3d_cp, intrinsic_mat, list(inliers))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(corner_storage)
    known_views = np.full(shape=frame_count, fill_value=None)

    if known_view_1 is None or known_view_2 is None:
        frame1, view1, frame2, view2 = init_frames(corner_storage, intrinsic_mat)
        known_views[frame1] = view1
        known_views[frame2] = view2
    else:
        known_views[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
        known_views[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
        frame1 = known_view_1[0]
        frame2 = known_view_2[0]

    last_frames = [frame1, frame2]
    known_corners = np.empty(shape=0, dtype=int)
    known_points3d = np.empty(shape=(0, 3))
    inliers = set()

    frames_queue = np.array([i for i in range(frame_count) if i != last_frames[-1] and i != last_frames[-2]])
    frames_queue = set(frames_queue)

    min_angle_param = 2

    for i in range(frame_count - 2):
        print(f"Frames left: {frame_count - 2 - i}")

        min_angle = min_angle_param

        if i == 0:
            min_angle = 0

        result, points3d, ids, _ = triangulate_2(last_frames, corner_storage, known_corners,
                                                 known_views, intrinsic_mat, min_angle, last_frames[-1])

        if result:
            print(f"New triangulated points: {len(points3d)}")
            known_points3d = np.vstack((known_points3d, points3d))
            known_corners = np.append(known_corners, ids)

        new_frame, new_view_mat = pnp_ransac(frames_queue, corner_storage, known_corners,
                                             known_points3d, intrinsic_mat, 15, inliers)
        frames_queue.remove(new_frame)
        known_views[new_frame] = new_view_mat
        last_frames.append(new_frame)

        print(f"Size of point cloud: {len(known_points3d)}")
        print("-------------------------------")

        if i % 20 == 0 and i != 0:
            refine_points_3d(known_points3d, known_corners, known_views, last_frames, corner_storage, intrinsic_mat)
            refine_views(last_frames, corner_storage, known_corners, known_points3d, intrinsic_mat, known_views, inliers)

    point_cloud_builder = PointCloudBuilder(known_corners,
                                            known_points3d)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        known_views,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, known_views))
    return poses, point_cloud


def init_frames(corner_storage: CornerStorage, intrinsic_mat):
    frame_count = len(corner_storage)
    best = 0
    res_frame1 = None
    res_frame2 = None
    res_view1 = None
    res_view2 = None
    step = 5

    known_views = np.full(shape=frame_count, fill_value=None)
    known_corners = np.empty(shape=0, dtype=int)

    for frame1 in range(0, frame_count, step):
        for frame2 in range(frame1 + step, frame_count, step):
            _, ids1, ids2 = np.intersect1d(corner_storage[frame1].ids.flatten(), corner_storage[frame2].ids.flatten(), return_indices=True)
            pts1 = corner_storage[frame1].points[ids1]
            pts2 = corner_storage[frame2].points[ids2]

            if len(pts1) < 50:
                break

            essential_mat, inliers = cv2.findEssentialMat(pts1, pts2, intrinsic_mat, cv2.RANSAC)

            _, homography_inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)

            if homography_inliers.sum() >= inliers.sum():
                continue

            _, r, t, _ = cv2.recoverPose(essential_mat, pts1, pts2, intrinsic_mat)
            view1 = eye3x4()
            view2 = np.hstack((r, t.reshape(-1, 1)))
            known_views[frame1] = view1
            known_views[frame2] = view2

            flag, _, _, angle = triangulate_2([frame1, frame2], corner_storage, known_corners,
                                              known_views, intrinsic_mat, 0, frame2)

            known_views[frame1] = None
            known_views[frame2] = None

            if flag and angle > best:
                best = angle
                res_frame1 = frame1
                res_frame2 = frame2
                res_view1 = view1
                res_view2 = view2

    return res_frame1, res_view1, res_frame2, res_view2


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
