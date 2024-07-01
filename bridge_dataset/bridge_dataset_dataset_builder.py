from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from bridge_dataset.conversion_utils import MultiThreadedDatasetBuilder
import json
import os
import cv2

gripper_pos_lookup = json.load(open("/nfs/kun2/users/oier/bridge_labeled_dataset_1.json", "r"))
depth_path = "/nfs/kun2/users/oier/bridge_depth"


def get_depth_point(depth_map, x, y, smooth=True):
    height, width = depth_map.shape
    if x >= height:
        # print("x is greater than height: ", x)
        x = height - 1
    elif x < 0:
        x = 0
    if y >= width:
        # print("y is greater than width: ", y)
        y = width - 1
    elif y < 0:
        y = 0
    if smooth:
        # Define the bounds of the neighborhood
        min_y = max(0, y - 1)
        max_y = min(width, y + 2)
        min_x = max(0, x - 1)
        max_x = min(height, x + 2)

        # Extract the neighborhood
        neighborhood = depth_map[min_x:max_x, min_y:max_y]

        # Calculate the average value of the neighborhood
        avg_value = np.mean(neighborhood)
        if np.isnan(avg_value):
            print("nan value found in depth map")
            print("x: ", x, " y: ", y)
        return avg_value
    else:
        return depth_map[x, y]


def compute_visual_trajectory(observation, depth_image, gripper_pos):
    assert len(observation) == depth_image.shape[0]
    assert len(observation) == len(gripper_pos)
    depth_tcp = []
    cumulative_depth_keypoints = []
    cumulative_keypoints = []
    traj_length = len(observation)
    max_img_depth = np.max(depth_image)
    min_img_depth = np.min(depth_image)
    tcp_3d = []
    for i in range(traj_length):
        depth_kp = get_depth_point(depth_image[i], int(gripper_pos[i][0]), int(gripper_pos[i][1]), smooth=True)
        depth_tcp.append(depth_kp)
        tcp_3d.append([gripper_pos[i][0], gripper_pos[i][1], depth_kp])

    for i in range(traj_length):
        pairs = [(gripper_pos[i], gripper_pos[i + 1]) for i in range(i, len(observation) - 1, 1)]
        depth_pairs = [(depth_tcp[i], depth_tcp[i + 1]) for i in range(i, len(observation) - 1, 1)]
        cumulative_depth_keypoints.append(depth_pairs)
        cumulative_keypoints.append(pairs)

    temp_color_list = [int(255 * (i / traj_length)) for i in range(traj_length)]
    count_idx = 0
    list_of_traj_imgs = []
    # print("total length of trajectory: ", traj_length)
    for i in range(traj_length):
        # print("processing image: ", i)
        current_image = observation[i]['images0']
        trajectory = cumulative_keypoints[i]
        # print("length of trajectory: ", len(trajectory))
        depth_traj = cumulative_depth_keypoints[i]
        for j, keypoints in enumerate(trajectory):
            depth_color = ((depth_traj[j][0] - min_img_depth) / (max_img_depth - min_img_depth) * 255.0)
            cv2.line(current_image, (int(keypoints[0][0]), int(keypoints[0][1])),
                     (int(keypoints[1][0]), int(keypoints[1][1])),
                     color=(0, depth_color, temp_color_list[count_idx:][j]), thickness=2)
        list_of_traj_imgs.append(current_image)
        count_idx += 1
        # from datetime import datetime
        #
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #
        # # Create the directory with the current timestamp
        # directory = os.path.join("/tmp", timestamp)
        # os.makedirs(directory, exist_ok=True)
        # cv2.imwrite(os.path.join(directory, f'{i}.jpg'), cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    # print("finished trajectory")
    return list_of_traj_imgs, tcp_3d


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(episode_path):
        # load raw data --> this should change for your dataset
        data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case

        for k, example in enumerate(data):
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            if episode_path in gripper_pos_lookup:
                gripper_pos = gripper_pos_lookup[episode_path][str(k)]['features']['gripper_position']
                #retrieve depth image
                meta_id = f'{k}__{episode_path}'
                meta_id = meta_id.replace('/', '\\')
                depth_file = os.path.join(depth_path, meta_id)
                if os.path.exists(depth_file):
                    depth_image = np.load(depth_file)
                    # print("loaded depth image shape: ", depth_image.shape)
                    list_traj_img, tcp_3d = compute_visual_trajectory(example['observations'], depth_image, gripper_pos)
                else:
                    print("depth image not found")
                    print("depth file: ", depth_file)

            else:
                print("gripper lookup not found")
            instruction = example['language'][0]
            if instruction:
                language_embedding = _embed([instruction])[0].numpy()
            else:
                language_embedding = np.zeros(512, dtype=np.float32)

            for i in range(len(example['observations'])):
                observation = {
                    'state': example['observations'][i]['state'].astype(np.float32),
                }
                for image_idx in range(4):
                    orig_key = f'images{image_idx}'
                    new_key = f'image_{image_idx}'
                    if orig_key in example['observations'][i]:
                        observation[new_key] = example['observations'][i][orig_key]
                    else:
                        observation[new_key] = np.zeros_like(example['observations'][i]['images0'])
                observation['visual_trajectory'] = list_traj_img[i]
                observation['depth'] = depth_image[i]
                observation['tcp_point_2d'] = np.array(gripper_pos[i], dtype=np.int32)
                observation['tcp_point_3d'] = np.array(tcp_3d[i], dtype=np.float32)
                episode.append({
                    'observation': observation,
                    'action': example['actions'][i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(example['observations']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(example['observations']) - 1),
                    'is_terminal': i == (len(example['observations']) - 1),
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': k,
                }
            }

            # mark dummy values
            for image_idx in range(4):
                orig_key = f'images{image_idx}'
                new_key = f'image_{image_idx}'
                sample['episode_metadata'][f'has_{new_key}'] = orig_key in example['observations']
            sample['episode_metadata']['has_language'] = bool(instruction)

            # if you want to skip an example for whatever reason, simply return None
            yield episode_path + str(k), sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        for id, sample in _parse_examples(sample):
            yield id, sample


class BridgeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    N_WORKERS = 1  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1  # number of paths converted & stored in memory before writing to disk
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_3': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        ),
                        'visual_trajectory': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Visual trajectory observation.',
                        ),
                        'depth': tfds.features.Tensor(
                            shape=(256, 256),
                            dtype=np.float32,
                            # encoding_format='jpeg',  # check of this is correct
                            doc='Main camera Depth observation.',
                        ),
                        'tcp_point_2d': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.int32,
                            doc='TCP 2d point.',
                        ),
                        'tcp_point_3d': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='TCP 3d point.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image0 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_1': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image1 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_2': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image2 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_3': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image3 exists in observation, otherwise dummy value.'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if language exists in observation, otherwise empty string.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        base_paths = ["/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256",
                      "/nfs/kun2/users/homer/datasets/bridge_data_all/scripted_numpy_256"]
        train_filenames, val_filenames = [], []
        for path in base_paths:
            for filename in glob.glob(f'{path}/**/*.npy', recursive=True):
                if '/train/out.npy' in filename:
                    train_filenames.append(filename)
                elif '/val/out.npy' in filename:
                    val_filenames.append(filename)
                else:
                    raise ValueError(filename)
        print(f"Converting {len(train_filenames)} training and {len(val_filenames)} validation files.")
        return {
            'train': train_filenames,
            'val': val_filenames,
        }
