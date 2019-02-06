import os.path
import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
import h5py

np.set_printoptions(threshold=np.nan)


def save_midi(piano_roll, filename):
    midi_output = pm.PrettyMIDI()
    guitar = pm.Instrument(program=25)
    bass = pm.Instrument(program=33)
    string = pm.Instrument(program=41)
    instruments = [guitar, bass, string]

    for i in range(piano_roll.shape[0]):
        for j in range(piano_roll.shape[1]):
            if piano_roll[i][j] < 0.8:
                piano_roll[i][j] = 0
            else:
                piano_roll[i][j] = 1

    piano_roll = np.array(np.hsplit(piano_roll, 3), dtype=np.int8)
    # print(piano_roll[0])
    # plt.imshow(piano_roll[0], cmap='gray')
    # plt.show()

    for i, track in enumerate(piano_roll):
        tick = 0
        while tick < track.shape[0]:
            if
        for t, tick in enumerate(track):
            for k, key in enumerate(tick):
                if key == 1:
                    key_hold = 1
                    start_time = t
                    end_time = start_time + key_hold

                    while t + key_hold < track.shape[0]:
                        print(t + key_hold)
                        if track[t + key_hold][key] == 1:
                            key_hold += 1
                            end_time += 1
                        else:
                            break
                    # zero_interval = 0
                    # while (t + key_hold) < track.shape[0]:
                    #     if track[t + key_hold][key] == 1:
                    #         track[t + key_hold][key] = 0
                    #         key_hold += 1
                    #         end_time += 1
                    #     elif track[t + key_hold][key] == 0 and zero_interval < 50 and end_time > 2:
                    #         key_hold += 1
                    #         end_time += 1
                    #         zero_interval += 1
                    #     else:
                    #         break

                    note = pm.Note(velocity=100,
                                   pitch=k,
                                   start=start_time,
                                   end=end_time)
                    instruments[i].notes.append(note)
        midi_output.instruments.append(instruments[i])

    midi_output.write(filename)


hf = h5py.File('../data/test_data.h5', 'r')
piano_roll_samples = np.array(hf.get('test'))
hf.close()

for p in range(piano_roll_samples.shape[0]):
    save_midi(piano_roll_samples[p], '../midi%d.mid' % p)
