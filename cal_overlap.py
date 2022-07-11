"""
reference:
https://github.com/prs-eth/OverlapPredator/blob/main/scripts/cal_overlap.py
"""
import argparse
import os
import glob
import pdb
import numpy as np
import open3d as o3d
import time
import json
from functools import partial
import multiprocessing as mp
import shutil

def load_info(info_txt):
    with open(info_txt, 'r') as f:
        content = f.readlines()
    line1 = content[1].rstrip().split('\t ')
    line2 = content[2].rstrip().split('\t ')
    line3 = content[3].rstrip().split('\t ')
    line4 = content[4].rstrip().split('\t ')

    T = np.vstack([line1, line2, line3, line4]).astype(np.float32)
    return T

def get_overlap_ratio(source, target, threshold=0.03):
    """
    Reference:
    https://github.com/prs-eth/OverlapPredator/blob/main/scripts/cal_overlap.py#L31
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_count = 0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if (count != 0):
            match_count += 1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio

def get_pair_info_per_scene(scene, data_root, ol_thres=0.3):
    tot_info = {}
    ply_files = glob.glob(os.path.join(data_root, scene, '*.ply'))
    info_files = glob.glob(os.path.join(data_root, scene, '*.info.txt'))
    assert len(ply_files) == len(info_files), f'make sure to fuse the fragments of {scene} correctly'
    ply_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=False)
    info_files.sort(key=lambda x: int(x.split('.')[-3].split('_')[-1]), reverse=False)

    for src_id in range(0, len(ply_files) - 1):
        ## load src
        src_ply = o3d.io.read_point_cloud(ply_files[src_id])
        src_trans = load_info(info_files[src_id])
        ## apply gt transform
        src_ply.transform(src_trans)
        src_ply = src_ply.voxel_down_sample(0.01)
        for tgt_id in range(src_id + 1, len(ply_files)):
            print(f'calculating {scene}\t{src_id}\t{tgt_id}')
            tgt_ply = o3d.io.read_point_cloud(ply_files[tgt_id])
            tgt_trans = load_info(info_files[tgt_id])
            tgt_ply.transform(tgt_trans)
            tgt_ply = tgt_ply.voxel_down_sample(0.01)
            ## cal overlap ratio
            ol_ratio = get_overlap_ratio(src_ply, tgt_ply)
            if ol_ratio > 0.3:
                name = f'{scene}/cloud_bin_{src_id}@{scene}/cloud_bin_{tgt_id}'
                tot_info[name] = ol_ratio
    json.dump(tot_info,
              open(os.path.join(data_root, f'{scene}.json'), 'w'),
              indent=4
              )

def get_pair_info(root, json_name, ol_thres=0.03):
    scenes = os.listdir(root)

    p = mp.Pool(processes=mp.cpu_count())
    p.map(partial(get_pair_info_per_scene, data_root=root, ol_thres=ol_thres), scenes)

    ### merge all json files into one json file
    json_files = glob.glob(os.path.join(root, '*.json'))
    assert len(json_files) == len(scenes), 'make sure all scene info have been generated!'
    tot_info_dict = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            scene_info = json.load(f)
        tot_info_dict.update(scene_info)
        os.remove(json_file)
    json.dump(tot_info_dict,
              open(json_name, 'w'),
              indent=4
              )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--ol_thres', type=float, default=0.3, help="only generate pair info for overlap ratio > ol_thres")
    parser.add_argument('--output_json', type=str, default='./ol_info.json')
    args = parser.parse_args()

    start = time.time()

    get_pair_info(root=args.root, json_name=args.output_json, ol_thres=args.ol_thres)

    end = time.time()
    print(f'{(end - start) / 3600:.2f}h for info generation....')