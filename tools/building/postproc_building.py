import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from osgeo import gdal, osr
from skimage.morphology import opening, square
from tqdm import tqdm

# isort: off
from spacenet8_model.utils.postproc_building import (
    get_flood_attributed_building_mask, polygonize_pred_mask,
    remove_small_polygons_and_simplify, geo_coords_to_image_coords)
# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foundation', required=True)
    parser.add_argument('--flood', required=True)
    parser.add_argument('--square_size',
                        type=int,
                        default=5)
    parser.add_argument('--min_area',
                        type=float,
                        default=5)
    parser.add_argument('--simplify_tolerance',
                        type=float,
                        default=0.75)
    parser.add_argument('--building_thresh',
                        type=float,
                        default=0.5)
    parser.add_argument('--flood_thresh',
                        type=float,
                        default=0.5)
    parser.add_argument('--flood_area_ratio',
                        type=float,
                        default=0.5)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def postprocess(pre_image_fn, args, aoi):
    # TODO:
    building_channel = 0
    building_flood_channel = 0
    flooded_building_label = 2  # same as sn-8 official baseline

    foundation_path = os.path.join(args.foundation, aoi, pre_image_fn)
    assert os.path.exists(foundation_path), foundation_path

    flood_path = os.path.join(args.flood, aoi, pre_image_fn)
    assert os.path.exists(flood_path)

    # load building mask from foundation mask
    ds = gdal.Open(foundation_path)
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    geotran = ds.GetGeoTransform()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    building_mask = ds.ReadAsArray()[building_channel].astype(float) / 255.0
    ds = None
    building_mask = (building_mask >= args.building_thresh).astype(np.uint8) * 255

    # morphological opening
    building_mask = opening(building_mask, square(args.square_size))

    # load building flood mask from flood mask
    flood_ds = gdal.Open(flood_path)
    flood_mask = flood_ds.ReadAsArray()[building_flood_channel].astype(float) / 255.0
    flood_label = np.zeros(shape=flood_mask.shape, dtype=np.uint8)
    flood_label[flood_mask >= args.flood_thresh] = flooded_building_label  # 2 flooded building, 0: others

    # assing flood label
    # 0: background, 1: non-flooded building, 2: flooded building
    label = get_flood_attributed_building_mask(building_mask, flood_label, perc_positive=args.flood_area_ratio)

    # convert mask to building polygons
    driver = gdal.GetDriverByName('MEM')
    outopen_ds = driver.Create('', ncols, nrows, 1, gdal.GDT_Byte)
    outopen_ds.SetGeoTransform(geotran)
    band = outopen_ds.GetRasterBand(1)
    outopen_ds.SetProjection(raster_srs.ExportToWkt())
    band.WriteArray(label)
    band.FlushCache()
    out_ds = polygonize_pred_mask(outopen_ds)

    # remove small polygons
    feats = remove_small_polygons_and_simplify(
        out_ds,
        area_threshold=args.min_area,
        simplify=True,
        simplify_tolerance=args.simplify_tolerance
    )

    image_id, _ = os.path.splitext(pre_image_fn)

    rows = []
    if len(feats) == 0: # no buildings detecting in the tile, write no prediction to submission
        # ['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']
        rows.append([image_id, 'Building', 'POLYGON EMPTY', 'False', 'Null', 'Null'])
    else:
        for f in feats:
            wkt_image_coords = geo_coords_to_image_coords(geotran, f['geometry'])
            flood_val = 'True' if f['properties']['mask_val'] == flooded_building_label else 'False'
            # ['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']
            rows.append([image_id, 'Building', wkt_image_coords, flood_val, 'Null', 'Null'])

    # submit without any road prediction
    # this line is removed when concat building and road dataframe
    rows.append([image_id, 'Road', 'LINESTRING EMPTY', 'False', 'Null', 'Null'])
    
    return rows


def process_aoi(args, aoi):
    paths = glob(os.path.join(args.foundation, aoi, '*.tif'))
    paths.sort()
    pre_image_fns = [os.path.basename(path) for path in paths]

    rows = []
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(pre_image_fns)) as pbar:
            for ret in pool.imap_unordered(partial(postprocess, args=args, aoi=aoi), pre_image_fns):
                rows.extend(ret)
                pbar.update()
    return rows


def main():
    args = parse_args()

    aois = [d for d in os.listdir(args.foundation) if os.path.isdir(os.path.join(args.foundation, d))]
    cols = ['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']
    rows = []
    for aoi in aois:
        print(f'processing {aoi} AOI')
        ret = process_aoi(args, aoi)
        rows.extend(ret)
    
    df = pd.DataFrame(rows, columns=cols)
    print(df.head(15))

    exp_foundation = os.path.basename(os.path.normpath(args.foundation)).replace('exp_', '')
    exp_flood = os.path.basename(os.path.normpath(args.flood)).replace('exp_', '')
    out_dir = f'exp_{exp_foundation}_{exp_flood}'
    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/building_submissions', out_dir)
    else:
        out_dir = os.path.join(args.artifact_dir, 'building_submissions', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'solution.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
