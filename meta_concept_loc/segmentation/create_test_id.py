import json
import numpy as np
import random
import os

key_frame_list = os.path.join('../../data/metadata', 'frame_test_id_dict.json')
key_frame = json.load(open(key_frame_list))
key_frame_list = list(key_frame.keys())

num_frames = 4
frame_id_all = []

for i in range(len(key_frame_list)): 
    frame_id_list = key_frame[key_frame_list[i]] 
    if len(frame_id_list) < num_frames:
        frame_id_list = [frame_id_list[i] for i in sorted(np.random.randint(len(frame_id_list), size=num_frames))]
    else:
        frame_id_list = [frame_id_list[i] for i in sorted(random.sample(range(len(frame_id_list)), num_frames))]
    # import pdb; pdb.set_trace()
    frame_id_all.extend(frame_id_list)

json.dump(frame_id_all, open('../../data/graph/frame_test_mask_id.json', 'w'))
