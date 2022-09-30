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
    corner_min_distance = 30
    max_corners = 500
    quality_level = 0.05
    block_size = 15
    corner_params = dict(qualityLevel=quality_level, minDistance=corner_min_distance, blockSize=block_size)
    lk_params = dict(winSize=(20, 20), maxLevel=3)

    prev_image = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(prev_image, max_corners, **corner_params)
    prev_frame_corners = FrameCorners(
        np.arange(corners.shape[0]),
        corners,
        np.ones(corners.shape[0]) * corner_min_distance
    )
    prev_min_eigen_val_max = cv2.cornerMinEigenVal(prev_image, block_size).max()
    builder.set_corners_at_frame(0, prev_frame_corners)

    id_cnt = corners.shape[0]
    for frame, image in enumerate(frame_sequence[1:], 1):
        next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(np.ubyte(prev_image * 255), np.ubyte(image * 255),
                                                         prev_frame_corners.points, None, **lk_params)
        ids = prev_frame_corners.ids.flatten()
        statuses = statuses.flatten()

        corner_min_eigen_val = cv2.cornerMinEigenVal(image, block_size).T
        flag = np.full(ids.shape[0], False)
        for i, (next_pt, status) in enumerate(zip(next_pts, statuses)):
            index = tuple(next_pt.round().astype(int))

            if status == 0 or index[0] >= image.shape[1] or index[1] >= image.shape[0]:
                continue

            if corner_min_eigen_val[index] > quality_level * prev_min_eigen_val_max:
                flag[i] = True

        ids = ids[flag]
        next_pts = next_pts[flag]

        mask = np.ubyte(np.ones(image.shape)) * 255

        for point in next_pts:
            mask = cv2.circle(mask, point.round().astype(int), corner_min_distance, 0, -1)

        next_pts = next_pts.reshape(next_pts.shape[0], 1, 2)

        if max_corners == 0:
            corners = cv2.goodFeaturesToTrack(image, 0, mask=mask, **corner_params)
        else:
            corners = cv2.goodFeaturesToTrack(image, max(max_corners - next_pts.shape[0], 1),
                                              mask=mask, **corner_params)

        if corners is not None:
            next_pts = np.append(next_pts, corners, axis=0)
            ids = np.append(ids, np.arange(id_cnt, id_cnt + corners.shape[0]))
            id_cnt += corners.shape[0]

        prev_frame_corners = FrameCorners(
            ids,
            next_pts,
            np.ones(ids.shape[0]) * corner_min_distance
        )
        prev_image = image
        prev_min_eigen_val_max = corner_min_eigen_val.max()

        builder.set_corners_at_frame(frame, prev_frame_corners)


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
