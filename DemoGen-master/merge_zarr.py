"""
Pack collected source demo(s) into a zarr file
"""

import argparse
import os
import numpy as np
import zarr
from termcolor import cprint
from os import listdir
from os.path import join
from tqdm import tqdm
import re
import pickle


def main(args):
    read_dir = os.path.join("data/source_demos", f"{args.exp_name}")
    save_dir = os.path.join("data/datasets/source", f"{args.exp_name}.zarr")

    print(f"Reading from {read_dir}")
    zarr_root = zarr.group(save_dir, overwrite=True)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    demos_list = [f for f in listdir(read_dir) if f.endswith('.pkl')]

    file_numbers = []
    for demo in demos_list:
        num = int(re.search(r'\d+', demo).group())
        file_numbers.append((num, demo))
    file_numbers.sort()

    state_arrays_ls, action_arrays_ls = [], []
    point_cloud_arrays_ls, episode_ends_arrays_ls =  [], []
    camera_metadata_ls = []

    ep_len = 0
    for num, demo in tqdm(file_numbers):
        with open(join(read_dir, demo), 'rb') as f:
            data = pickle.load(f)
            episode_len = (data["agent_pos"]).shape[0]
            state_arrays_ls.append(data["agent_pos"])
            action_arrays_ls.append(data["action"])
            point_cloud_arrays_ls.append(data["point_cloud"])
            camera_metadata_ls.append(data.get("camera_metadata"))
            
        ep_len += episode_len
        episode_ends_arrays_ls.append(ep_len)
    
    state_arrays = np.concatenate(state_arrays_ls, axis=0)
    #print(state_arrays)
    action_arrays = np.concatenate(action_arrays_ls, axis=0)
    point_cloud_arrays = np.concatenate(point_cloud_arrays_ls, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays_ls)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])

    zarr_data.create_dataset('agent_pos', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    if len(camera_metadata_ls) == len(file_numbers) and all(md is not None for md in camera_metadata_ls):
        camera_group = zarr_meta.create_group('camera')
        camera_group.create_dataset(
            'fovy_deg',
            data=np.asarray([md['camera_fovy_deg'] for md in camera_metadata_ls], dtype=np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'policy_depth_shape_hw',
            data=np.stack([md['policy_depth_shape_hw'] for md in camera_metadata_ls]).astype(np.int32),
            dtype='int32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'sam_rgb_shape_hw',
            data=np.stack([md['sam_rgb_shape_hw'] for md in camera_metadata_ls]).astype(np.int32),
            dtype='int32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'resized_obs_shape_hw',
            data=np.stack([md['resized_obs_shape_hw'] for md in camera_metadata_ls]).astype(np.int32),
            dtype='int32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'workspace_bounds',
            data=np.stack([md['workspace_bounds'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'intrinsics_policy',
            data=np.stack([md['intrinsics_policy'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'intrinsics_sam',
            data=np.stack([md['intrinsics_sam'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'camera_position_world',
            data=np.stack([md['extrinsics']['camera_position_world'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'camera_to_world',
            data=np.stack([md['extrinsics']['camera_to_world'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.create_dataset(
            'world_to_camera',
            data=np.stack([md['extrinsics']['world_to_camera'] for md in camera_metadata_ls]).astype(np.float32),
            dtype='float32',
            overwrite=True,
            compressor=compressor,
        )
        camera_group.attrs['camera_names'] = [md['camera_name'] for md in camera_metadata_ls]
        camera_group.attrs['sam_source_image_paths'] = [md['sam_source_image_path'] for md in camera_metadata_ls]
    else:
        cprint('Camera metadata not found in all source demos, skip saving camera meta to zarr.', 'yellow')

    cprint(f'-'*50, 'cyan')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    if args.save_h5:
        # save to hdf5
        import h5py
        save_dir = os.path.join("data", f"source{args.exp_name}.hdf5")
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('agent_pos', data=state_arrays, compression='gzip')
            f.create_dataset('action', data=action_arrays, compression='gzip')
            f.create_dataset('point_cloud', data=point_cloud_arrays, compression='gzip')
            f.create_dataset('episode_ends', data=episode_ends_arrays, compression='gzip')

        cprint(f'Saved hdf5 file to {save_dir}', 'green')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('exp_name', type=str)
    args.add_argument('--save_h5', action='store_true')
    args = args.parse_args()

    main(args)