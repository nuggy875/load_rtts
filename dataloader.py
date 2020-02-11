import torch
import os
import os.path as osp
import xml.etree.ElementTree as ET
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image

FILTER_DIFFICULT = True


def parse_annotation(path):
    assert(os.path.exists(path)), \
        'Annotation: {} does not exist'.format(path)
    tree = ET.parse(path)
    objs = tree.findall('object')
    boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1.0
        y1 = float(bbox.find('ymin').text) - 1.0
        x2 = float(bbox.find('xmax').text) - 1.0
        y2 = float(bbox.find('ymax').text) - 1.0
        box = [x1, y1, x2, y2]
        cls = obj.find('name').text.lower().lower()
        difficult = int(obj.find('difficult').text) == 1
        if FILTER_DIFFICULT:
            if not difficult:
                boxes.append({'cls': cls, 'box': box})
        else:
            boxes.append({'cls': cls, 'box': box})
    return boxes


class RTTS_Dataset(data.Dataset):
    def __init__(self, dataset_dir, format, train, size):
        super(RTTS_Dataset, self).__init__()
        self.dest_files = osp.join(dataset_dir, 'ImageSets', 'Main', 'test.txt')
        self.im_dir = osp.join(dataset_dir, 'JPEGImages')
        self.anno_dir = osp.join(dataset_dir, 'Annotations')
        with open(self.dest_files, 'r') as f:
            names = f.readlines()
        self.names = [im.strip() for im in names]
        self.im_files = [osp.join(self.im_dir, img) for img in self.names]
        self.format = format

    def __getitem__(self, index):
        print('>>INDEX:', index)
        img_name = self.names[index]
        img_path = osp.join(self.im_dir, (img_name+self.format))
        anno_path = osp.join(self.anno_dir, (img_name+'.xml'))
        bbox = parse_annotation(anno_path)
        # haze=Image.open(img_path+self.format)

        return 

    def __len__(self):
        return len(self.names)

    def collate_fn(self, batch):
        images = list()

        for b in batch:
            images.append(b)

        images = torch.stack(images, dim=0)
        return images


if __name__ == "__main__":
    input_size = 800
    batch_size = 1
        
    data_path = '/workspace/data'
    dataset = RTTS_Dataset(data_path+'/RESIDE/RTTS', format='.png', train=True, size=input_size)
    RTTS_train_loader=DataLoader(dataset=dataset, batch_size=batch_size, 
                                collate_fn=dataset.collate_fn, shuffle=False)
    #iters_per_epoch = int(len(RTTS_train_loader)/batch_size)
    iters_per_epoch = int(len(RTTS_train_loader))
    data_iter = iter(RTTS_train_loader)
    for step in range(iters_per_epoch):
        data = next(data_iter)