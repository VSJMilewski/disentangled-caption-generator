from PIL import Image
from torchvision import models, transforms
import torch
import pickle
import numpy as np
import os
from random import shuffle
import random

class data():
    def __init__(self, base_path_images, subset_images, annotations, max_length, processor, filename, start, end):
        self.base_path_images = base_path_images
        self.subset_images = subset_images
        self.max_length = max_length
        self.START = start
        self.END = end
        if os.path.isfile(filename):
            self.samples = pickle.load(open(filename, 'rb'))
        else:
            self.samples = self.create_samples(annotations, processor)
        pickle.dump(self.samples,open(filename,'wb'))

    def create_samples(self, annotations, processor):
        samples = dict()
        for image in self.subset_images:
            captions = []
            for line in annotations:
                sp = line.split('\t')
                if sp[0][:-2] == image:
                    caption = [self.START] + sp[1].split() + [self.END]
                    captions.append(np.array([processor.w2i[w] for w in caption]))
                    if len(captions) == 5:
                        break
            samples[image] = captions
        return samples


def batch_generator(data, batch_size, image_transform, device, seed=42):
    keys = list(data.samples.keys())
    random.Random(seed).shuffle(keys)
    current_id = 0

    while True:
        if current_id == len(keys):
            break

        batch_images = []
        batch_captions = []
        lengths = []
        image_names = []
        for i in range(batch_size):
            if current_id == len(keys):
                batch_size = len(batch_captions)
                break
            captions = data.samples[keys[current_id]]
            for c in captions:
                lengths.append(len(c))
            batch_captions += captions
            image_names.append(keys[current_id])
            image = Image.open(data.base_path_images+keys[current_id])
            batch_images += [image_transform(image)]*5
            current_id += 1

        batch_cap = np.zeros([len(batch_captions), data.max_length])
        for j,cap in enumerate(batch_captions):
            batch_cap[j, :len(cap)] = cap

        batch_im = torch.stack(batch_images, dim=0).to(device)
        batch_cap = torch.from_numpy(batch_cap).type(torch.LongTensor).to(device)
        lengths = torch.FloatTensor(lengths).to(device)

        yield (batch_im, batch_cap, lengths, image_names)


def batch_generator_dev(data, batch_size, image_transform, device, seed=42):
    keys = list(data.samples.keys())
    current_id = 0

    while True:
        if current_id == len(keys):
            break

        batch_images = []
        image_names = []
        for i in range(batch_size):
            if current_id == len(keys):
                batch_size = len(batch_images)
                break
            image_names.append(keys[current_id])
            image = Image.open(data.base_path_images+keys[current_id])
            batch_images += [image_transform(image)]
            current_id += 1

        batch_im = torch.stack(batch_images, dim=0).to(device)

        yield (batch_im, image_names)
