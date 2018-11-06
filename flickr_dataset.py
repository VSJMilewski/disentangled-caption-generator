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
            # data.to_pickle(filename)
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
        caption = np.zeros((1, self.max_len))
        caption_len = len(self.caption_arr[item])
        caption[0, :caption_len] = self.caption_arr[item]
        caption = torch.from_numpy(caption).type(torch.LongTensor)
        return image, caption, image_name, caption_len

    def __len__(self):
        return len(self.image_arr)

    @staticmethod
    def create_dataframe(annotations, subset, processor, unique=False):
        data = []
        past_images = []
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
                    past_images.append(image)
                    caption = [processor.w2i[processor.START]] + \
                              [processor.w2i[w.lower()] for w in sp[1].split()] + \
                              [processor.w2i[processor.END]]
                    if len(caption) > max_len:
                        max_len = len(caption)
                    data.append((image, np.array(caption)))
        print(max_len)
        return pd.DataFrame(data)

#     def create_samples(self, annotations, processor):
#         samples = dict()
#         for image in self.subset_images:
#             captions = []
#             for line in annotations:
#                 sp = line.split('\t')
#                 if sp[0][:-2] == image:
#                     caption = [self.START] + sp[1].split() + [self.END]
#                     captions.append(np.array([processor.w2i[w.lower()] if w not in (self.START, self.END)
#                                               else processor.w2i[w]
#                                               for w in caption]))
#                     # there are 5 captions for each image
#                     if len(captions) == 5:
#                         break
#             samples[image] = captions
#         return samples
#
#
#
# def batch_generator(data_, batch_size, image_transform, device):
#     keys = list(data_.samples.keys())
#     data_.rnd.shuffle(keys)
#     current_id = 0
#
#     while True:
#         if current_id == len(keys):
#             break
#
#         batch_images = []
#         batch_captions = []
#         lengths = []
#         image_names = []
#         for i in range(batch_size):
#             if current_id == len(keys):
#                 batch_size = len(batch_captions)
#                 break
#             captions = data_.samples[keys[current_id]]
#             for c in captions:
#                 lengths.append(len(c))
#             batch_captions += captions
#             image_names.append(keys[current_id])
#             image = Image.open(data_.base_path_images + keys[current_id])
#             batch_images += [image_transform(image)]*5
#             current_id += 1
#
#         batch_cap = np.zeros([len(batch_captions), max(lengths) + 2])
#         for j,cap in enumerate(batch_captions):
#             batch_cap[j, :len(cap)] = cap
#
#         batch_im = torch.stack(batch_images, dim=0).to(device)
#         batch_cap = torch.from_numpy(batch_cap).type(torch.LongTensor).to(device)
#         lengths = torch.FloatTensor(lengths).to(device)
#
#         yield (batch_im, batch_cap, lengths, image_names)
#
#
# def batch_generator_dev(data_, batch_size, image_transform, device, seed=42):
#     keys = list(data_.samples.keys())
#     current_id = 0
#
#     while True:
#         if current_id == len(keys):
#             break
#
#         batch_images = []
#         image_names = []
#         for i in range(batch_size):
#             if current_id == len(keys):
#                 batch_size = len(batch_images)
#                 break
#             image_names.append(keys[current_id])
#             image = Image.open(data_.base_path_images + keys[current_id])
#             batch_images += [image_transform(image)]
#             current_id += 1
#
#         batch_im = torch.stack(batch_images, dim=0).to(device)
#
#         yield (batch_im, image_names)
