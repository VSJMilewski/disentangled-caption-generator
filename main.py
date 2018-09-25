# loadbars to track the run/speed
from tqdm import tqdm, trange

# numpy for arrays/matrices/mathematical stuff
import numpy as np

# nltk for tokenizer
from nltk.tokenize import wordpunct_tokenize

# torch for the NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

# torch tools for data processing
from torch.utils.data import DataLoader
import pycocotools #cocoAPI

# torchvision for the image dataset and image processing
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from torchvision import models

#coco captions evaluation
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap

# packages for plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import skimage.io as io

# additional stuff
import pickle
from collections import Counter
from collections import defaultdict
import os
import time

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# import other files
from model import *
from vocab_flickr8k import *
from caption_eval.evaluations_function import *
from flickr8k_data_processor import *

#test if there is a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hyper parameters
PAD = '<PAD>'
START = '<START>'
END = '<END>'
UNK = '<UNK>'

vocab_size = 30000
max_sentence_length = 60

learning_rate = 1e-3
max_epochs = 1000
min_epochs = 0
batch_size = 25  # 5 images per sample, 13x5=65, 25x5=125

embedding_size = 512

patience = 10

resize_size = int(299 / 224 * 256)
crop_size = 299
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
transform_eval = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

#setup paths
prediction_file = 'output/dev_baseline.pred'
last_epoch_file = './output/last_flickr8k_baseline_model.pkl'
best_epoch_file = './output/best_flickr8k_baseline_model.pkl'
train_images_file = './data/flickr8k/Flickr_8k.trainImages.txt'
dev_images_file = './data/flickr8k/Flickr_8k.devImages.txt'
test_images_file = './data/flickr8k/Flickr_8k.testImages.txt'
base_path_images = './data/flickr8k/Flicker8k_Dataset/'
reference_file = './data/flickr8k/Flickr8k_references.dev.json'
captions_file = './data/flickr8k/Flickr8k.token.txt'
train_vocab_file = './train_flickr8k_vocab_' + str(vocab_size) + '.pkl'
dev_vocab_file = './dev_flickr8k_vocab_' + str(vocab_size) + '.pkl'
train_data_file = './data_flickr8k_train.pkl'
dev_data_file = './data_flickr8k_dev.pkl'
test_data_file = './data_flickr8k_test.pkl'


# setup data stuff
print('reading data files...')
start = time.time()
train_images = None
dev_images = None
test_images = None
annotations = None
with open(train_images_file) as f:
    train_images = f.read().splitlines()
with open(dev_images_file) as f:
    dev_images = f.read().splitlines()
with open(test_images_file) as f:
    test_images = f.read().splitlines()
with open(captions_file) as f:
    annotations = f.read().splitlines()
end = time.time()
print("time opening files: " + str(end - start))
print('create/open vocabulary')
start = time.time()
dev_processor = DataProcessor(annotations, dev_images, filename= dev_vocab_file , vocab_size=vocab_size)
dev_processor.save()
processor = DataProcessor(annotations, train_images, filename= train_vocab_file, vocab_size=vocab_size)
processor.save()
end = time.time()
print("open vocab: " + str(end - start))

print('create data processing objects...')
start = time.time()
train_data = data(base_path_images, train_images, annotations, max_sentence_length, processor, train_data_file, START, END)
dev_data = data(base_path_images, dev_images, annotations, max_sentence_length, processor, dev_data_file, START, END)
test_data = data(base_path_images, test_images, annotations, max_sentence_length, processor, test_data_file , START, END)
end = time.time()
print("data processor: " + str(end - start))

# create the models
print('create model...')
start = time.time()
caption_model = CaptionModel(embedding_size, processor.vocab_size, device).to(device)
caption_model.train(True)  # probably not needed. better to be safe
params = list(caption_model.encoder.inception.fc.parameters()) + list(caption_model.decoder.parameters())
opt = Adam(params, lr=learning_rate)
end = time.time()
print("model created: " + str(end - start))

# variables for training
losses = []
avg_losses = []
loss_current_ind = 0
scores = []
best_bleu = -1
best_epoch = -1
number_up = 0


def validation_step(model, batch_size):
    model.eval()
    enc = model.encoder.to(device)
    dec = model.decoder.to(device)

    predicted_sentences = dict()
    for image, image_name in batch_generator_dev(dev_data, batch_size, transform_eval, device):
        # Encode
        h0 = enc(image)

        # prepare decoder initial hidden state
        h0 = h0.unsqueeze(0)
        c0 = torch.zeros(h0.shape).to(device)
        hidden_state = (h0, c0)

        # Decode
        start_token = torch.LongTensor([processor.w2i[START]]).to(device)
        predicted_words = []
        prediction = start_token.view(1, 1)
        for w_idx in range(max_sentence_length):
            prediction, hidden_state = dec(prediction, hidden_state)

            index_predicted_word = np.argmax(prediction.detach().cpu().numpy(), axis=2)[0][0]
            predicted_word = processor.i2w[index_predicted_word]
            predicted_words.append(predicted_word)

            if predicted_word == END:
                break
            prediction = torch.LongTensor([index_predicted_word]).view(1, 1).to(device)
        predicted_sentences[image_name[0]] = predicted_words
        del start_token
        del prediction

    # perform validation
    with open(prediction_file, 'w', encoding='utf-8') as f:
        for im, p in predicted_sentences.items():
            if p[-1] == END:
                p = p[:-1]
            f.write(im + '\t' + ' '.join(p) + '\n')

    score = evaluate(prediction_file, reference_file)
    scores.append(score)
    torch.save(caption_model.state_dict(), last_epoch_file)
    caption_model.train()


# loop over number of epochs
print('training...')
print('{:8s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:7s}\n{}'.format(
    'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 75))
for epoch in range(max_epochs):
    start = time.time()
    # loop over all the training batches
    for i_batch, batch in enumerate(batch_generator(train_data, batch_size, transform_train, device)):
        opt.zero_grad()
        image, caption, caption_lengths, _ = batch
        image = image.to(device)
        caption = caption.to(device)
        caption_lengths = caption_lengths.to(device)
        loss = caption_model(image, caption, caption_lengths)
        loss.backward()
        losses.append(float(loss))
        opt.step()

    # create validation result file
    validation_step(caption_model, 1)
    if score['Bleu_4'] <= best_bleu:
        number_up += 1
        if number_up > patience and epoch > min_epochs:
            print('=' * 75 + '\nFinished training!')
            break
    else:
        number_up = 0
        torch.save(caption_model.state_dict(), best_epoch_file)
        best_bleu = scores[-1]['Bleu_4']
        best_epoch = epoch

    end = time.time()
    print('{:5d}   \t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:7.3f}'.format(
        epoch, scores[-1]['Bleu_1'], scores[-1]['Bleu_2'], scores[-1]['Bleu_3'], scores[-1]['Bleu_4'],
        scores[-1]['METEOR'], scores[-1]['ROUGE_L'], scores[-1]['CIDEr'], end - start))

print('\n' + '=' * 75 + '\n\t\tBest Epoch: ')
print('{:5d}   \t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}'.format(
    best_epoch, scores[best_epoch]['Bleu_1'], scores[best_epoch]['Bleu_2'], scores[best_epoch]['Bleu_3'],
    scores[best_epoch]['Bleu_4'], scores[best_epoch]['METEOR'], scores[best_epoch]['ROUGE_L'],
    scores[best_epoch]['CIDEr']))
print('=' * 75 + '\n')

pickle.dump(scores, open('./output/scores_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))
pickle.dump(losses, open('./output/losses_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))
pickle.dump(avg_losses, open('./output/avg_glosses_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))

torch.save(caption_model.state_dict(), './output/last_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch))
