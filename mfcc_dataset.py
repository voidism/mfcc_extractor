import os
import h5py
import torch
from torch.utils.data import Dataset
import tqdm

class MfccDataset(Dataset):
    def __init__(self, file_path, to_tensor=True):
        self.to_tensor = to_tensor

        if not os.path.isfile(file_path):
            raise OSError("No such file '{}'".format(file_path))

        self.h5_file = h5py.File(file_path, 'r+')

        if not os.path.exists(file_path + ".index"):
            print("Index file not found, wait a while for it to iterating over hdf5 dataset...")
            fw = open(file_path + ".index", 'w')
            for line in tqdm.tqdm(self.h5_file.keys()):
                fw.write(line + '\n')
            fw.close()

        index_file = open(file_path + ".index", 'r')
        self.id_list = index_file.read().strip().split('\n')
        self.id_dict = dict([(self.id_list[i], i) for i in range(len(self.id_list))])

    def __getitem__(self, index):
        if not self.to_tensor:
            return self.h5_file[self.id_list[index]][...]
        else:
            return torch.from_numpy(self.h5_file[self.id_list[index]][...]).float()

    def get_item_from_id(self, item_id):
        if not self.to_tensor:
            return self.h5_file[item_id][...]
        else:
            return torch.from_numpy(self.h5_file[item_id][...]).float()

    def __len__(self):
        return len(self.id_list)

# ======================= Usage Example =========================

if __name__ == '__main__':
    # create dataset from hdf5 file
    dataset = MfccDataset("MFCC-960-librispeech.hdf5")

    # define padding function for torch tensor
    def padding_fn(tensor_list):
        lens = [len(x) for x in tensor_list]
        max_len = max(lens)
        # you can sort tensors by length here, if you want to use pack sequence for RNNs before pytorch-1.1
        ret = []
        for array in tensor_list:
            ret.append(torch.cat((array, torch.zeros(max_len - len(array), array.shape[-1], device=array.device, dtype=array.dtype)), dim=0).unsqueeze(0))
        return torch.cat(ret, dim=0), lens
    
    # create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=padding_fn)

    # iterate
    for batch in dataloader:
        print("batch shape:", batch[0].shape)
        print("sequence length", batch[1])
        break