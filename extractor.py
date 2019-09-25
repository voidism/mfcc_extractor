import librosa
import os
import h5py
import numpy as np
from operator import itemgetter
import tqdm
# NOTE: there are warnings for MFCC extraction due to librosa's issue
import warnings
warnings.filterwarnings("ignore")

# Source: https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/preprocess.py
# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file,feature='mfcc',dim=13, cmvn=True, delta=True, delta_delta=True,
                    window_size=25, stride=10,save_feature=None):
    y, sr = librosa.load(input_file,sr=None)
    ws = int(sr*0.001*window_size)
    st = int(sr*0.001*stride)
    if feature == 'fbank': # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                    n_fft=ws, hop_length=st)
        feat = np.log(feat+1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: '+feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0],order=2))
    feat = np.concatenate(feat,axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat,0,1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat,0,1).astype('float32')

def walk_all_mfcc(dirname, hdf5name, feature, dim, window_size, stride, cmvn, delta, ddelta):
    h5f = h5py.File(hdf5name, "w")
    all_path = list(os.walk(dirname))
    for dirPath, dirNames, fileNames in tqdm.tqdm(all_path):
        if len(fileNames)!=0:
            for f in tqdm.tqdm(fileNames):
                if not (".flac" in f or ".wav" in f):
                    # print("Skip", f)
                    continue
                key, ftype = f.split('.')
                array = extract_feature(os.path.join(dirPath, f), feature, dim, cmvn, delta, ddelta, window_size, stride)
                dset = h5f.create_dataset(key, array.shape, dtype=array.dtype)
                dset[...] = array
            print(dirPath, "done with %d utterances."%len(fileNames), flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='extract MFCC features from all audio files in a directory.')
    parser.add_argument("--dir", type=str, required=True, help="directory containing all audio files, files can be putted nested.")
    parser.add_argument("--to", type=str, required=True, help="hdf5 file name to store the extracted MFCC features.")
    parser.add_argument("--feature", type=str, default='mfcc', help="feature type, `fbank` or `mfcc`. (default mfcc)")
    parser.add_argument("--dim", type=int, default=13, help="dimension of feature, 13 for mfcc and 20 for fbank before adding deltas. (default 13)")
    parser.add_argument("--window_size", type=int, default=25, help="window size for FFT (ms)")
    parser.add_argument("--stride", type=int, default=10, help="window stride for FFT")
    parser.add_argument("--cmvn", type=int, default=1, help="set to 1 to apply CMVN on feature, else 0. (default 1)")
    parser.add_argument("--delta", type=int, default=1, help="set to 1 to add delta, else 0. (default 1)")
    parser.add_argument("--d_delta", type=int, default=1, help="set to 1 to add delta-delta, else 0. (default 1)")
    args = parser.parse_args()
    walk_all_mfcc(args.dir, args.to, args.feature, args.dim, args.window_size, args.stride, args.cmvn, args.delta, args.d_delta)

