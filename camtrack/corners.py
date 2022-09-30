#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli, filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corner_min_distance = 50
    max_corners = 200
    corner_params = dict(qualityLevel=0.01, minDistance=corner_min_distance)
    lk_params = dict(winSize=(20, 20), maxLevel=3)
    eigen_val_multiplier = 0.9

    image_0 = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(image_0, max_corners, **corner_params)
    prev_corner_frames = FrameCorners(
        np.arange(corners.shape[0]),
        corners,
        np.ones(corners.shape[0]) * corner_min_distance
    )
    prev_corner_min_eigen_val = cv2.cornerMinEigenVal(image_0, corner_min_distance).T
    builder.set_corners_at_frame(0, prev_corner_frames)

    id_cnt = corners.shape[0]
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(np.ubyte(image_0 * 255), np.ubyte(image_1 * 255),
                                                         prev_corner_frames.points, None, **lk_params)
        ids = prev_corner_frames.ids.reshape(prev_corner_frames.ids.shape[0])
        statuses = statuses.reshape(statuses.shape[0])

        corner_min_eigen_val = cv2.cornerMinEigenVal(image_1, corner_min_distance).T
        flag = np.full(ids.shape[0], False)
        for i, (next_pt, pt, status) in enumerate(zip(next_pts, prev_corner_frames.points, statuses)):
            next_index = tuple(next_pt.round().astype(int))
            index = tuple(pt.round().astype(int))

            if status == 0 or next_index[0] >= image_1.shape[1] or next_index[1] >= image_1.shape[0]:
                continue

            if corner_min_eigen_val[next_index] > eigen_val_multiplier * prev_corner_min_eigen_val[index]:
                flag[i] = True

        ids = ids[flag]
        next_pts = next_pts[flag]

        mask = np.ubyte(np.ones(image_1.shape)) * 255

        for point in next_pts:
            mask = cv2.circle(mask, point.round().astype(int), corner_min_distance, 0, -1)

        next_pts = next_pts.reshape(next_pts.shape[0], 1, 2)

        if max_corners == 0:
            corners_1 = cv2.goodFeaturesToTrack(image_1, 0, mask=mask, **corner_params)
        else:
            corners_1 = cv2.goodFeaturesToTrack(image_1, max(max_corners - next_pts.shape[0], 1), mask=mask,
                                                **corner_params)

        if corners_1 is not None:
            next_pts = np.append(next_pts, corners_1, axis=0)
            ids = np.append(ids, np.arange(id_cnt, id_cnt + corners_1.shape[0]))

        corner_frames = FrameCorners(
            ids,
            next_pts,
            np.ones(ids.shape[0]) * corner_min_distance
        )

        image_0 = image_1
        builder.set_corners_at_frame(frame, corner_frames)
        prev_corner_frames = corner_frames
        prev_corner_min_eigen_val = corner_min_eigen_val
    kek = 8


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
