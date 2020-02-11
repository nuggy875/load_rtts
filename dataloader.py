import os
import os.path as osp
import torch.utils.data as data
from torch.utils.data import DataLoader

class RTTS_Dataset(data.Dataset):
    def __init__(self, dataset_dir, train, size):
        super(RTTS_Dataset, self).__init__()
        self.dest_files = os.path.join(dataset_dir, 'ImageSets', 'Main', 'test.txt')
        self.im_dir = os.path.join(dataset_dir, 'JPEGImages')
        self.anno_dir = os.path.join(dataset_dir, 'Annotations')

    def __getitem__(self, index):
        print('test')

    def __len__(self):
        return len(self.im_dir)


data_path = '/workspace/data'

RTTS_train_loader=DataLoader(dataset=RESIDE_Dataset(data_path+'/RESIDE/RTTS', train=True, size=input_size), batch_size=batch_size, shuffle=True)