import argparse
import math
import os
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--flood', required=True)
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def write_road_submission_shapefile(df, out_shapefile):
    df = df.reset_index()  # make sure indexes pair with number of rows

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # Create the output shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(out_shapefile)
    out_layer = ds.CreateLayer(out_shapefile[:-4], srs, ogr.wkbLineString)
    
    fieldnames = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo', 'length_m']
    
    field_name = ogr.FieldDefn('ImageId', ogr.OFTString)
    field_name.SetWidth(100)
    out_layer.CreateField(field_name)
    ob = ogr.FieldDefn('Object', ogr.OFTString)
    ob.SetWidth(10)
    out_layer.CreateField(ob)
    flooded = ogr.FieldDefn('Flooded', ogr.OFTString)
    flooded.SetWidth(5)
    out_layer.CreateField(flooded)
    pix = ogr.FieldDefn('WKT_Pix', ogr.OFTString)
    pix.SetWidth(255)
    out_layer.CreateField(pix)
    geo = ogr.FieldDefn('WKT_Geo', ogr.OFTString)
    geo.SetWidth(255)
    out_layer.CreateField(geo)
    out_layer.CreateField(ogr.FieldDefn('length_m', ogr.OFTReal))
    #out_layer.CreateField(ogr.FieldDefn('travel_t_s', ogr.OFTReal))

    # Create the feature and set values
    featureDefn = out_layer.GetLayerDefn()
    
    for index, row in df.iterrows():
        
        outFeature = ogr.Feature(featureDefn)
        for j in fieldnames:
            if j == "travel_time_s":
                pass
                #outFeature.SetField('travel_t_s', row[j])
            else:
                outFeature.SetField(j, row[j])
        
        geom = ogr.CreateGeometryFromWkt(row["WKT_Geo"])
        outFeature.SetGeometry(geom)
        out_layer.CreateFeature(outFeature)
        outFeature = None
    ds = None


def pkl_dir_to_wkt(pkl_dir,
                   weight_keys=['length', 'travel_time_s'],
                   verbose=False):
    """
    Create submission wkt from directory full of graph pickles
    """
    wkt_list = []

    pkl_list = sorted([z for z in os.listdir(pkl_dir) if z.endswith('.gpickle')])
    for i, pkl_name in enumerate(tqdm(pkl_list)):
        # print(pkl_name)
        G = nx.read_gpickle(os.path.join(pkl_dir, pkl_name))
        
        # ensure an undirected graph
        if verbose:
            print(i, "/", len(pkl_list), "num G.nodes:", len(G.nodes()))

        #name_root = pkl_name.replace('PS-RGB_', '').replace('PS-MS_', '').split('.')[0]
        name_root = pkl_name.replace("_roadspeedpred", '').split('.')[0]

        # AOI_root = 'AOI' + pkl_name.split('AOI')[-1]
        # name_root = AOI_root.split('.')[0].replace('PS-RGB_', '')
        if verbose:
            print("name_root:", name_root)
        
        # if empty, still add to submission
        if len(G.nodes()) == 0:
            wkt_item_root = [name_root, 'Road', 'LINESTRING EMPTY', 'LINESTRING EMPTY', 'False']
            if len(weight_keys) > 0:
                weights = ['Null' for w in weight_keys]
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

        # extract geometry pix wkt, save to list
        seen_edges = set([])
        for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
            # make sure we haven't already seen this edge
            if (u, v) in seen_edges or (v, u) in seen_edges:
                if verbose:
                    print(u, v, "already catalogued!")
                continue
            else:
                seen_edges.add((u, v))
                seen_edges.add((v, u))
            geom_pix = attr_dict['geometry_pix']
            if type(geom_pix) != str:
                geom_pix_wkt = attr_dict['geometry_pix'].wkt
            else:
                geom_pix_wkt = geom_pix
            
            geom_geo = attr_dict["geometry_utm_wkt"].wkt
            #if type(geom_geo) != str:
            #    geom_geo_wkt = attr_dict["geometry_wkt"].wkt
            #else:
            #    geom_geo_wkt = geom_geo
            # geometry_wkt is in a UTM coordinate system..
            geom = ogr.CreateGeometryFromWkt(geom_geo)

            targetsrs = osr.SpatialReference()
            targetsrs.ImportFromEPSG(4326)

            utm_zone = attr_dict['utm_zone']
            source = osr.SpatialReference() # the input dataset is in wgs84
            source.ImportFromProj4(f'+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs')
            transform_to_utm = osr.CoordinateTransformation(source, targetsrs)
            geom.Transform(transform_to_utm)
            geom_geo_wkt = geom.ExportToWkt()

            # check edge lnegth
            #if attr_dict['length'] > 5000:
            #    print("Edge too long!, u,v,data:", u,v,attr_dict)
            #    return
            
            if verbose:
                print(i, "/", len(G.edges()), "u, v:", u, v)
                print("  attr_dict:", attr_dict)
                print("  geom_pix_wkt:", geom_pix_wkt)
                print("  geom_geo_wkt:", geom_geo_wkt)

            wkt_item_root = [name_root, 'Road', geom_pix_wkt, geom_geo_wkt, 'False']
            if len(weight_keys) > 0:
                weights = [attr_dict[w] for w in weight_keys]
                if verbose:
                    print("  weights:", weights)
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

    if verbose:
        print("wkt_list:", wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded'] + weight_keys
    else:
        cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded']

    # use 'length_m' and 'travel_time_s' instead?
    cols_new = []
    for z in cols:
        if z == 'length':
            cols_new.append('length_m')
        elif z == 'travel_time':
            cols_new.append('travel_time_s')
        else:
            cols_new.append(z)
    cols = cols_new
    # cols = [z.replace('length', 'length_m') for z in cols]
    # cols = [z.replace('travel_time', 'travel_time_s') for z in cols]
    # print("cols:", cols)

    df = pd.DataFrame(wkt_list, columns=cols)
    # print("df:", df)
    # save
    #if len(output_csv_path) > 0:
    #    df.to_csv(output_csv_path, index=False)
    return df


def insert_flood_pred(flood_pred_dir, df, road_flood_channel, flood_thresh):
    flood_road_label = 4  # same as sn-8 baseline (any positive int should be okay)

    dy=2
    dx=2
    cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded', 'length_m']
    out_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        imageid = row["ImageId"]
        
        flood_pred_filename = os.path.join(flood_pred_dir, f"{imageid}.tif")
        assert(os.path.exists(flood_pred_filename)), "flood prediction file for this linestring doesn't exist"

        ds = gdal.Open(flood_pred_filename)
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize

        # XXX: kimura modified sn-8 baseline
        flood_mask = ds.ReadAsArray()[road_flood_channel].astype(float) / 255.0
        flood_arr = np.zeros(shape=flood_mask.shape, dtype=np.uint8)
        flood_arr[flood_mask >= flood_thresh] = flood_road_label  # 4: flooded road, 0: others

        if row["WKT_Pix"] != "LINESTRING EMPTY":
            geom = ogr.CreateGeometryFromWkt(row["WKT_Pix"])        

            nums = [] # flood prediction vals
            for i in range(0, geom.GetPointCount()-1):
                pt1 = geom.GetPoint(i)
                pt2 = geom.GetPoint(i+1)
                dist = math.ceil(math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))
                x0, y0 = pt1[0], pt1[1]
                x1, y1 = pt2[0], pt2[1]
                x, y = np.linspace(x0, x1, dist).astype(int), np.linspace(y0, y1, dist).astype(int)

                for i in range(len(x)):
                    top = max(0, y[i]-dy)
                    bot = min(nrows-1, y[i]+dy)
                    left = max(0, x[i]-dx)
                    right = min(ncols-1, x[i]+dx)
                    nums.extend(flood_arr[top:bot,left:right].flatten())

            currow = row
            maxval = np.argmax(np.bincount(nums))
            if maxval == flood_road_label:
                currow["Flooded"] = "True"

        out_rows.append([currow[k] for k in list(currow.keys())])

    df = pd.DataFrame(out_rows, columns=cols)
    return df


def process_aoi(args, aoi):
    # TODO:
    road_flood_channel = 2
    flood_thresh = 0.5
    
    graph_dir = os.path.join(args.graph, aoi)
    if not os.path.exists(graph_dir):
        print(f'graph_dir does not exists for {aoi}')
        return None

    print('graph -> wkt ..')
    df = pkl_dir_to_wkt(
        graph_dir,
        weight_keys=['length'],
        verbose=False
    )

    print('inserting flood attribute ..')
    flood_dir = os.path.join(args.flood, aoi)
    df = insert_flood_pred(
        flood_dir,
        df,
        road_flood_channel,
        flood_thresh
    )
    
    return df


def add_empty_rows(args, df, cols):
    image_ids = []
    aois = [d for d in os.listdir(args.flood) if os.path.isdir(os.path.join(args.flood, d))]
    for aoi in aois:
        paths = glob(os.path.join(args.flood, aoi, '*.tif'))
        ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        image_ids.extend(ids)
    image_ids.sort()

    empty_rows = []
    for image_id in image_ids:
        # submit without any building prediction
        # this line is removed when concat building and road dataframe
        empty_rows.append([image_id, 'Building', 'POLYGON EMPTY', 'POLYGON EMPTY', 'False', 'Null'])

        if image_id not in list(df.ImageId.unique()):
            # add images where no road was detected
            empty_rows.append([
                image_id, 'Road', 'LINESTRING EMPTY', 'LINESTRING EMPTY', 'False', 'Null'
            ])
    df = df.append(pd.DataFrame(empty_rows, columns=cols))

    return df


def main():
    args = parse_args()

    cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded', 'length_m']  # 'travel_time_s' will be added later
    df = pd.DataFrame(columns=cols)
    aois = [d for d in os.listdir(args.flood) if os.path.isdir(os.path.join(args.flood, d))]
    for aoi in aois:
        print(f'processing {aoi} AOI')
        ret = process_aoi(args, aoi)
        if ret is not None:
            df = df.append(ret)

    image_ids = []
    for aoi in aois:
        paths = glob(os.path.join(args.flood, aoi, '*.tif'))
        ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        image_ids.extend(ids)
    image_ids.sort()

    df = add_empty_rows(args, df, cols)

    df['travel_time_s'] = 'Null'
    df = df.drop(columns='WKT_Geo')

    exp_foundation = os.path.basename(os.path.normpath(args.graph)).replace('exp_', '')
    exp_flood = os.path.basename(os.path.normpath(args.flood)).replace('exp_', '')
    out_dir = f'exp_{exp_foundation}_{exp_flood}'
    out_dir = os.path.join(args.artifact_dir, 'road_submissions', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'solution.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()