#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import scipy.ndimage

import dask.array as da

import dask_image.ndmeasure


@pytest.mark.parametrize(
    "chunksize, size, ndim", [
        (5, 10, 2),
        (5, 10, 3),
        (5, 10, 4),
    ]
)
def test_apply_additive_label_offsets(chunksize, size, ndim):

    np.random.seed(0)
    im = np.random.randint(0, 100, [size] * ndim) > 60
    im_da = da.from_array(im, chunks=[chunksize] * ndim)

    labels = scipy.ndimage.label(im)[0]

    labels_mb = da.map_blocks(
        lambda x: scipy.ndimage.label(x)[0],
        im_da,
        dtype=labels.dtype)

    labels_mb_offset_nonmax = dask_image.ndmeasure._utils._label._apply_additive_label_offsets(
        labels_mb,
        use_max_labels=False,
    )

    labels_mb_offset_max = dask_image.ndmeasure._utils._label._apply_additive_label_offsets(
        labels_mb,
        use_max_labels=True,
    )

    assert(len(np.array(np.unique(labels_mb_offset_nonmax))) ==\
        len(np.array(np.unique(labels_mb_offset_max))))

    assert(len(np.array(np.unique(labels_mb_offset_max))) >\
        len(np.array(np.unique(labels_mb))))

    assert(len(np.array(np.unique(labels_mb_offset_max))) >=\
        len(np.array(np.unique(labels))))


def test_process_labels_at_tile_boundaries():

    labels_gt = np.array([0, 1, 1, 2, 2, 0])
    labels_gt_da = da.from_array(labels_gt, chunks=(3, ))

    # with depth=0 the ccs are joined and the ground truth labels are not recovered
    labels_joined_depth0_da = dask_image.ndmeasure.process_labels_at_tile_boundaries(
        labels_gt_da,
        input_block_labels_are_approx_sequential=True,
        input_block_labels_are_globally_unique=True,
        overlap_depth=0)
    labels_joined_depth0 = labels_joined_depth0_da.compute(scheduler='single-threaded')

    assert(np.allclose(labels_joined_depth0, [0, 1, 1, 1, 1, 0]))

    # with depth=1 the ground truth labels are recovered
    labels_gt_da_overlap_depth1 = da.overlap.overlap(labels_gt_da,
        {dim: 1 for dim in range(labels_gt.ndim)}, boundary='none')
    labels_joined_depth1_da = dask_image.ndmeasure.process_labels_at_tile_boundaries(
        labels_gt_da_overlap_depth1,
        input_block_labels_are_approx_sequential=True,
        input_block_labels_are_globally_unique=True,
        overlap_depth=1)
    labels_joined_depth1 = labels_joined_depth1_da.compute(scheduler='single-threaded')

    assert(np.allclose(labels_joined_depth1, [0, 1, 1, 2, 2, 0]))
