import os.path
import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
import h5py


def save_midi(piano_roll, filename):
    midi_output = pm.PrettyMIDI()
    guitar = pm.Instrument(program=25)
    bass = pm.Instrument(program=33)
    string = pm.Instrument(program=41)
    instruments = [guitar, bass, string]

    for i in range(piano_roll.shape[0]):
        for j in range(piano_roll.shape[1]):
            if piano_roll[i][j] < 0.9:
                piano_roll[i][j] = 0
            else:
                piano_roll[i][j] = 1

    piano_roll = np.hsplit(piano_roll, 3)

    for i, track in enumerate(piano_roll):
        for t, tick in enumerate(track):
            for k, key in enumerate(tick):
                if key == 1:
                    key_hold = 1
                    start_time = t
                    while (t + key_hold) <= track.shape[0] and track[t + key_hold][key] == 1:
                        track[t + key_hold][key] = 0
                        key_hold += 1
                    end_time = start_time + key_hold

                    note = pm.Note(velocity=100,
                                   pitch=k,
                                   start=start_time * 0.15,
                                   end=end_time * 0.15)
                    instruments[i].notes.append(note)
        midi_output.instruments.append(instruments[i])

    midi_output.write(filename)
