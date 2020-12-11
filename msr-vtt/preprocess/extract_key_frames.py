import json
import os
import math
import random
from tqdm import tqdm


#######################################################
########## GVD video processing methods ###############
########## sample frames every 0.5s     ###############
#######################################################


import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
import shutil
import multiprocessing
 
def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x
 
def extract_key_frames(params):

    USE_TOP_ORDER = False

    video_id = params
    key_ids = []

    frame_list = os.listdir(os.path.join(split_path, video_id))
    frame_count = len(frame_list)

    curr_frame = None
    prev_frame = None 
    frame_diffs = []
    frames = []

    for i in range(frame_count):
        frame_id = i+1
        fream_path = os.path.join(split_path, video_id, str(frame_id).zfill(6)+'.jpg')
        frame = cv2.imread(fream_path)
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        
        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
    

    #smoothing window size
    key_frame_num = 6
    len_window = max(int(frame_count / key_frame_num), 3) 
    NUM_TOP_FRAMES = key_frame_num

    # compute keyframe
    keyframe_id_set = set()
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)   
    # if USE_LOCAL_MAXIMA:
    #     # print("Using Local Maxima")
    #     diff_array = np.array(frame_diffs)
    #     sm_diff_array = smooth(diff_array, len_window)
    #     frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    #     for i in frame_indexes:
    #         # keyframe_id_set.add(frames[i - 1].id)
    #         frame_id = i+1
    #         key_ids.append(os.path.join(video_id, str(frame_id).zfill(6)+'.jpg'))
    #     if key_ids == []:
    #         USE_TOP_ORDER = True

    # if USE_TOP_ORDER:
    #     # sort the list in descending order
    #     frames.sort(key=operator.attrgetter("diff"), reverse=True)

    #     for keyframe in frames[:NUM_TOP_FRAMES]:
    #             key_ids.append(os.path.join(video_id, str(keyframe.id).zfill(6)+'.jpg'))

    if USE_LOCAL_MAXIMA:
        # print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i, idx in enumerate(frame_indexes):
            if i == 0:
                sample_id = random.sample(range(0, idx), 1)[0] + 1
            else:
                sample_id = random.sample(range(frame_indexes[i-1], idx), 1)[0] + 1

            frame_id = idx+1
            assert str(frames[sample_id].id).zfill(6)+'.jpg' in frame_list
            assert str(frames[frame_id].id).zfill(6)+'.jpg' in frame_list
            key_ids.append(os.path.join(video_id, str(frames[sample_id].id).zfill(6)+'.jpg'))
            key_ids.append(os.path.join(video_id, str(frames[frame_id].id).zfill(6)+'.jpg'))
            # import pdb; pdb.set_trace()
        if len(frame_indexes) == 0:
            # sort the list in descending order
            # frames.sort(key=operator.attrgetter("diff"), reverse=True)
            sorted_idx = sorted(range(len(frames)), key=lambda k: frames[k].diff, reverse=True)
            frame_id = np.where(np.argsort(sorted_idx)==0)[0][0]
            if frame_id == 0:
                sample_id = random.sample(range(frame_id+1, len(frames)), 1)[0]
            else:
                sample_id = random.sample(range(0, frame_id), 1)[0]

            assert str(frames[sample_id].id).zfill(6)+'.jpg' in frame_list
            assert str(frames[frame_id].id).zfill(6)+'.jpg' in frame_list
            key_ids.append(os.path.join(video_id, str(frames[sample_id].id).zfill(6)+'.jpg'))
            key_ids.append(os.path.join(video_id, str(frames[frame_id].id).zfill(6)+'.jpg'))

    key_ids = list(set(key_ids))
    key_ids.sort()

    #######################################################
    ###### Write the selected frames to the disk ##########
    #######################################################

    # demo_imgs_path = 'demo_imgs'
    # original_frames_path = demo_imgs_path+'/original_frames/'
    # key_frames_path = demo_imgs_path+'/key_frames/'
    # if os.path.exists(demo_imgs_path):
    #     shutil.rmtree(demo_imgs_path)
    #     os.makedirs(original_frames_path)
    #     os.makedirs(key_frames_path)
    # else:
    #     os.makedirs(original_frames_path)
    #     os.makedirs(key_frames_path)


    # keyframe_id_set = frame_indexes
    # for i in range(frame_count):
    #     frame_id = i+1
    #     frame_path = os.path.join(split_path, video_id, str(frame_id).zfill(6)+'.jpg')
    #     frame = cv2.imread(frame_path)
    #     cv2.imwrite(original_frames_path + str(frame_id).zfill(6)+'.jpg', frame)

    #     frame_name = os.path.join(video_id, str(frame_id).zfill(6)+'.jpg')
    #     # import pdb; pdb.set_trace()
    #     if frame_name in key_ids:
    #         cv2.imwrite(key_frames_path + str(frame_id).zfill(6)+'.jpg', frame)

    return key_ids

if __name__ == "__main__":
    # print(sys.executable)
    #Setting fixed threshold criteria
    USE_THRESH = False
    #fixed threshold value
    THRESH = 0.6
    #Setting fixed threshold criteria
    USE_TOP_ORDER = False
    #Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    #Number of top sorted frames
    NUM_TOP_FRAMES = 50
    
    processed_filename = []

    video_path = '../msrvtt_frames/'
    split_path = video_path
    video_ids = os.listdir(split_path)

    # for i, video_id in tqdm(enumerate(video_ids)):
    #     key_ids = extract_key_frames(video_id)
    #     import pdb; pdb.set_trace()

    pool = multiprocessing.Pool(20)
    for key_ids in tqdm(pool.imap_unordered(extract_key_frames, video_ids), total=len(video_ids)):
        processed_filename.extend(key_ids)
    pool.close()
    pool.join()

    print("The length: " + str(len(processed_filename)))
    json.dump(processed_filename, open('../files/processed_filename.json', 'w'))


