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
    Correspondences,
    TriangulationParameters,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    compute_reprojection_errors
)


def triangulate(frames, corner_storage, known_corners, known_views, intrinsic_mat, min_angle):
    known_corners = np.sort(known_corners)
    max_new_points = 0
    result_points3d = None
    result_ids = None

    for frame1, frame2 in itertools.product(frames, frames):
        if frame1 == frame2:
            continue

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
            flag = np.isin(correspondences.ids, ids)
            error1 = compute_reprojection_errors(points3d, correspondences.points_1[flag],
                                                 intrinsic_mat @ known_views[frame1]).mean()

            error2 = compute_reprojection_errors(points3d, correspondences.points_2[flag],
                                                 intrinsic_mat @ known_views[frame2]).mean()

            error = max(error1, error2)

            if len(points3d) == 20:
                continue

            max_new_points = len(points3d)
            result_points3d = points3d
            result_ids = ids

    if result_points3d is not None:
        print(f"max new points {max_new_points}, error {error}")
        return True, result_points3d, result_ids
    else:
        return False, None, None


def pnp(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, inliers):
    max_inliers = -1
    result_frame = None

    for frame in frames_queue:
        flag = np.isin(corner_storage[frame].ids.flatten(), inliers)

        if flag.sum() > max_inliers:
            max_inliers = flag.sum()
            result_frame = frame

    flag = np.isin(corner_storage[result_frame].ids.flatten(), inliers)
    flag2 = np.isin(known_corners, corner_storage[result_frame].ids.flatten()[flag])

    new_view = cv2.solvePnP(known_points3d[flag2], corner_storage[result_frame].points[flag],
                            intrinsic_mat, None)
    view_mat = rodrigues_and_translation_to_view_mat3x4(new_view[1], new_view[2])

    error = compute_reprojection_errors(known_points3d[flag2], corner_storage[result_frame].points[flag], intrinsic_mat @ view_mat).mean()
    print(f"regular pnp frame: {result_frame}, inliers: {max_inliers}, isok: {new_view[0]}, error: {error}")

    if error > 1000:
        print(f'bad frame {result_frame}')

    return result_frame, view_mat


def pnp_ransac(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, min_corners, inliers):
    for frame in frames_queue:
        flag = np.isin(corner_storage[frame].ids.flatten(), known_corners)

        if flag.sum() < min_corners:
            continue

        flag2 = np.isin(known_corners, corner_storage[frame].ids.flatten()[flag])
        new_view = cv2.solvePnPRansac(known_points3d[flag2], corner_storage[frame].points[flag],
                                      intrinsic_mat, None, reprojectionError=1.0, confidence=0.999, iterationsCount=10000)

        if not new_view[0]:
            continue

        view_mat = rodrigues_and_translation_to_view_mat3x4(new_view[1], new_view[2])

        error = compute_reprojection_errors(known_points3d[flag2], corner_storage[frame].points[flag],
                                            intrinsic_mat @ view_mat).mean()

        if error > 10.0:
            print(f"big ransac error: {error}")
            continue

        inliers.update(corner_storage[frame].ids[flag][new_view[3].flatten()].flatten())
        print(f"ransac pnp frame: {frame}, error: {error}")
        return frame, view_mat
        return pnp(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, list(inliers))



    return pnp(frames_queue, corner_storage, known_corners, known_points3d, intrinsic_mat, list(inliers))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)

    known_views = np.full(shape=frame_count, fill_value=None)
    known_views[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    known_views[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    last_frames = [known_view_1[0], known_view_2[0]]
    known_corners = np.empty(shape=0, dtype=int)
    known_points3d = np.empty(shape=(0, 3))
    inliers = set()

    frames_queue = np.array([i for i in range(frame_count) if i != last_frames[-1] and i != last_frames[-2]])
    frames_queue = set(frames_queue)

    min_angle_param = 3

    for i in range(frame_count - 2):
        print(len(frames_queue))

        min_angle = min_angle_param

        if i == 0:
            min_angle = 0

        result, points3d, ids = triangulate(last_frames, corner_storage, known_corners,
                                            known_views, intrinsic_mat, min_angle)

        if result:
            known_points3d = np.vstack((known_points3d, points3d))
            known_corners = np.append(known_corners, ids)

        new_frame, new_view_mat = pnp_ransac(frames_queue, corner_storage, known_corners,
                                             known_points3d, intrinsic_mat, 20, inliers)
        frames_queue.remove(new_frame)
        known_views[new_frame] = new_view_mat
        last_frames.append(new_frame)

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


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
