import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class FlickrDataset(Dataset):

    def __init__(self, base_path, annotations, subset_images, processor, filename, transform, max_len, unique=False):
        self.base_path = base_path
        self.transform = transform
        self.max_len = max_len
        if os.path.isfile(filename):
            data = pd.read_pickle(filename)
        else:
            data = self.create_dataframe(annotations, subset_images, processor, unique)
            data.to_pickle(filename)
        # First column contains the image paths
        self.image_arr = np.asarray(data.iloc[:, 0])
        # Second column is the labels
        self.caption_arr = np.asarray(data.iloc[:, 1])

    def __getitem__(self, item):
        # process image
        image_name = self.image_arr[item]
        image = Image.open(self.base_path + image_name)
        image = self.transform(image)

        # process caption
        caption = np.zeros((self.max_len))
        caption_len = len(self.caption_arr[item])
        caption[:caption_len] = self.caption_arr[item]
        caption = torch.from_numpy(caption).type(torch.LongTensor)
        return image, caption, image_name, torch.FloatTensor([caption_len])

    def __len__(self):
        return len(self.image_arr)

    @staticmethod
    def create_dataframe(annotations, subset, processor, unique=False):
        data = []
        past_images = set()
        max_len = 0
        for line in annotations:
            sp = line.split('\t')
            image = sp[0][:-2]
            if not unique:
                if image in subset:
                    caption = [processor.w2i[processor.START]] + \
                              [processor.w2i[w.lower()] for w in sp[1].split()] + \
                              [processor.w2i[processor.END]]
                    if len(caption) > max_len:
                        max_len = len(caption)
                    data.append((image, np.array(caption)))
            else:
                if image in subset and image not in past_images:
                    past_images.add(image)
                    caption = [processor.w2i[processor.START]] + \
                              [processor.w2i[w.lower()] for w in sp[1].split()] + \
                              [processor.w2i[processor.END]]
                    if len(caption) > max_len:
                        max_len = len(caption)
                    data.append((image, np.array(caption)))
        print(max_len)
        return pd.DataFrame(data)

