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
    corner_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=3)
    lk_params = dict(winSize=(20, 20), maxLevel=3)
    barrier = 10.0

    image_0 = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(image_0, **corner_params)
    prev_corner_frames = FrameCorners(
        np.arange(corners.shape[0]),
        corners,
        np.ones(corners.shape[0]) * 30
    )
    builder.set_corners_at_frame(0, prev_corner_frames)

    id_cnt = corners.shape[0]
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(np.ubyte(image_0 * 255), np.ubyte(image_1 * 255),
                                                         prev_corner_frames.points, None, **lk_params)

        corners_1 = cv2.goodFeaturesToTrack(image_1, **corner_params)

        ids = np.ones(corners_1.shape[0], int) * -1
        for pt, corner_id, status in zip(next_pts, prev_corner_frames.ids, statuses):
            if status == 0:
                continue

            distances = np.sqrt((corners_1[:, 0, 0] - pt[0]) ** 2 + (corners_1[:, 0, 1] - pt[1]) ** 2)
            index = distances.argmin()
            min_distance = distances.min()
            if min_distance < barrier:
                ids[index] = corner_id

        for i in range(len(ids)):
            if ids[i] == -1:
                ids[i] = id_cnt
                id_cnt += 1

        corner_frames = FrameCorners(
            ids,
            corners_1,
            np.ones(corners_1.shape[0]) * 30
        )
        image_0 = image_1
        builder.set_corners_at_frame(frame, corner_frames)
        prev_corner_frames = corner_frames


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
