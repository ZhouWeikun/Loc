#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raster row/col <-> projected coordinate helpers.

The project datasets use GDAL-style geotransforms:
    Xgeo = GT[0] + col * GT[1] + row * GT[2]
    Ygeo = GT[3] + col * GT[4] + row * GT[5]

These helpers keep the existing call signatures used across the codebase.
"""

from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer


def _to_np(x):
    return np.asarray(x, dtype=np.float64)


@lru_cache(maxsize=32)
def _get_transformer(source_epsg_code, target_epsg_code):
    source = CRS.from_epsg(int(source_epsg_code))
    target = CRS.from_epsg(int(target_epsg_code))
    return Transformer.from_crs(source, target, always_xy=True)


def _transform_xy(x, y, source_epsg_code, target_epsg_code):
    if int(source_epsg_code) == int(target_epsg_code):
        return x, y
    transformer = _get_transformer(source_epsg_code, target_epsg_code)
    x_new, y_new = transformer.transform(x, y)
    return _to_np(x_new), _to_np(y_new)


def _apply_geotransform(cols, rows, geotransform):
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in geotransform]
    cols = _to_np(cols)
    rows = _to_np(rows)
    x = gt0 + cols * gt1 + rows * gt2
    y = gt3 + cols * gt4 + rows * gt5
    return x, y


def _invert_geotransform(x, y, geotransform):
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in geotransform]
    x = _to_np(x)
    y = _to_np(y)

    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-12:
        raise ValueError(f"Invalid geotransform with near-zero determinant: {geotransform}")

    dx = x - gt0
    dy = y - gt3
    cols = (gt5 * dx - gt2 * dy) / det
    rows = (-gt4 * dx + gt1 * dy) / det
    return rows, cols


def georc_to_raster_rc(geoy, geox, source_epsg_code, target_epsg_code, target_geotransform):
    """
    Convert projected coordinates to raster row/col in the target raster CRS.

    Returns:
        rows, cols
    """
    geox = _to_np(geox)
    geoy = _to_np(geoy)
    x_tgt, y_tgt = _transform_xy(geox, geoy, source_epsg_code, target_epsg_code)
    rows, cols = _invert_geotransform(x_tgt, y_tgt, target_geotransform)
    return rows.astype(np.float64), cols.astype(np.float64)


def raster_rc_to_georc(rows, cols, source_geotransform, source_epsg_code, target_epsg_code):
    """
    Convert raster row/col to projected coordinates.

    Returns:
        geoy, geox
    """
    rows = _to_np(rows)
    cols = _to_np(cols)
    x_src, y_src = _apply_geotransform(cols, rows, source_geotransform)
    x_tgt, y_tgt = _transform_xy(x_src, y_src, source_epsg_code, target_epsg_code)
    return y_tgt.astype(np.float64), x_tgt.astype(np.float64)


def rc_offset_to_meters(offset_col, offset_row, geotransform, epsg_code=None):
    """
    Convert a raster row/col offset into projected x/y offsets.

    For the datasets in this repo, the raster CRS is a projected CRS with meter units,
    so the linear geotransform terms already represent meter offsets.
    """
    gt1 = float(geotransform[1])
    gt2 = float(geotransform[2])
    gt4 = float(geotransform[4])
    gt5 = float(geotransform[5])
    offset_col = _to_np(offset_col)
    offset_row = _to_np(offset_row)
    offset_x_m = gt1 * offset_col + gt2 * offset_row
    offset_y_m = gt4 * offset_col + gt5 * offset_row
    return offset_x_m, offset_y_m
