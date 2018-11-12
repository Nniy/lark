import os.path
import pypianoroll
import numpy as np
import h5py


# np.set_printoptions(threshold=np.nan)

def get_filenames(root, suffix=None):
    file_path_list = []
    if not suffix:
        suffix = ''
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in [f for f in filenames if f.endswith(suffix)]:
            file_path_list.append(os.path.join(dirpath, filename))
    return file_path_list


def load_npz(filepath):
    multi_tracks = []
    pm = pypianoroll.load(filepath).to_pretty_midi()
    resolution = pm.resolution
    time = pm.get_end_time()
    tick = pm.time_to_tick(time * 8 / resolution)
    tracks = pm.instruments
    for track in tracks:
        piano_roll = track.get_piano_roll(fs=tick / time)
        if tick < 640:
            padded_pr = np.pad(piano_roll, ((0, 0), (0, 640 - len(piano_roll[0]))), 'constant').T
        else:
            padded_pr = np.pad(piano_roll, ((0, 0), (0, tick - len(piano_roll[0]))), 'constant').T
        multi_tracks.append(padded_pr[:640])

    concat_pr = multi_tracks[0]
    for i in multi_tracks[1:]:
        concat_pr = np.concatenate((concat_pr, i), axis=1)
    return concat_pr


hf = h5py.File('../data/data.h5', 'w')
list_pr = get_filenames("../data/lpd_5_cleansed/")

train = []
for pr in list_pr:
    train.append(load_npz(pr))
hf.create_dataset('train', data=np.array(train))
hf.close()
