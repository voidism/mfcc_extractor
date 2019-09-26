# MFCC extractor
Simple one-line scripts to extract reliable MFCC features with librosa and store in HDF5 format file.

## Requirements
- librosa
- tqdm
- h5py

install them by:
```bash
pip install -r requirements.txt
```

## Usage
```bash
usage: extractor.py [-h] --dir DIR --to TO [--feature FEATURE] [--dim DIM]
                    [--window_size WINDOW_SIZE] [--stride STRIDE]
                    [--cmvn CMVN] [--delta DELTA] [--d_delta D_DELTA]

extract MFCC features from all audio files in a directory.

optional arguments:
  -h, --help            show this help message and exit
  --dir DIR             directory containing all audio files, files can be
                        putted nested.
  --to TO               hdf5 file name to store the extracted MFCC features.
  --feature FEATURE     feature type, `fbank` or `mfcc`. (default mfcc)
  --dim DIM             dimension of feature, 13 for mfcc and 20 for fbank
                        before adding deltas. (default 13)
  --window_size WINDOW_SIZE
                        window size for FFT (ms)
  --stride STRIDE       window stride for FFT
  --cmvn CMVN           set to 1 to apply CMVN on feature, else 0. (default 1)
  --delta DELTA         set to 1 to add delta, else 0. (default 1)
  --d_delta D_DELTA     set to 1 to add delta-delta, else 0. (default 1)
```

### Extract features
- download the audio files into one directory (the audio files can be placed in many subdirectories or mixed with other non-audio files)
- e.g. download LibriSpeech audios from http://www.openslr.org/12 and unzip all file to one dir `LibriSpeech/`
- run 
```bash
python extractor.py --dir LibriSpeech --to MFCC-960-librispeech.hdf5
```
all mfcc features will be stored in `MFCC-960-librispeech.hdf5`.

### Download the extracted hdf5 file for Librispeech
I have extract mfcc feature for:
- [train-clean-100.tar.gz](http://www.openslr.org/resources/12/train-clean-100.tar.gz)
- [train-clean-360.tar.gz](http://www.openslr.org/resources/12/train-clean-360.tar.gz)
- [train-other-500.tar.gz](http://www.openslr.org/resources/12/train-other-500.tar.gz)

with default arguments use `extractor.py` and upload it to google drive.

You can download it by:
```bash
bash download_librispeech-960hr.sh
```
> warning: the file size is about 50G.

## Create dataset and dataloader for extracted hdf5 file

```python
from mfcc_dataset import MfccDataset
# create dataset from hdf5 file
# it will generate index file for the first time. please wait a while.
dataset = MfccDataset("MFCC-960-librispeech.hdf5")

# define padding function for torch tensor
def padding_fn(tensor_list):
    lens = [len(x) for x in tensor_list]
    max_len = max(lens)
    # you can sort tensors by length here, if you want to use pack sequence for RNNs for old torch version
    ret = []
    for array in tensor_list:
        ret.append(torch.cat((array, torch.zeros(max_len - len(array), array.shape[-1], device=array.device, dtype=array.dtype)), dim=0).unsqueeze(0))
    return torch.cat(ret, dim=0), lens

# create dataloader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=padding_fn)

# iterate over batches
for batch in dataloader:
    print("batch shape:", batch[0].shape)
    print("sequence length", batch[1])
    break
```

## Reference
I borrow the code here to extract reliable mfcc features.  

https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/preprocess.py
