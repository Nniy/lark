import os.path
import pypianoroll
import numpy as np
import matplotlib.pyplot as plt
import h5py

np.set_printoptions(threshold=np.nan)


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


def gbs_piano_roll(filepath):
    pm = pypianoroll.load(filepath).to_pretty_midi()
    instruments = pm.instruments
    # g = Guitar; b = Bass; s = String.
    gbs = []
    for instrument in instruments[2:]:
        if len(instrument.notes) == 0:
            return None
        else:
            gbs.append(instrument.notes)

    piano_roll_length = pm.time_to_tick(pm.get_end_time()) * 4 // pm.resolution
    piano_roll = np.zeros((128 * 3, piano_roll_length), dtype=np.int8)

    for i, instrument in enumerate(gbs):
        for note in instrument:
            start = pm.time_to_tick(note.start) * 4 // pm.resolution
            end = pm.time_to_tick(note.end) * 4 // pm.resolution
            for tick in range(end - start):
                piano_roll[(i * 128) + note.pitch][start + tick] = 1
    # plt.imshow(piano_roll, cmap='gray')
    # plt.show()
    return piano_roll


no = 0
data = []
filenames = get_filenames("../data/lpd_5_cleansed/")
hf = h5py.File('../data/gbs_data.h5', 'w')

for file in filenames:
    p = gbs_piano_roll(file)
    if p is not None:
        p = p.T
        num_phrase = len(p) // 384
        for i in range(num_phrase):
            data.append(p[(i * 384):((i + 1) * 384)])
            if len(data) == 10000:
                dataset = np.array(data, dtype=np.int8)
                hf.create_dataset('data%d' % no, data=dataset, dtype=np.int8)
                print('Done: ', no)
                no += 1
                data = []

hf.close()

