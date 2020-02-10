import torch.utils.data as data
from torch.utils.data import DataLoader

class RTTS_Dataset(data.Dataset):
    def __init__(self, path, train, size):
        super(RTTS_Dataset, self).__init__()

    def __getitem__(self, index):
        print('test')



data_path = '/workspace/data'

RTTS_train_loader=DataLoader(dataset=RESIDE_Dataset(data_path+'/RESIDE/RTTS', train=True, size=input_size), batch_size=batch_size, shuffle=True)