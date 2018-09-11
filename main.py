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
from torch.optim import SGD

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
#device = torch.device('cpu') # uncomment if cuda does not work
print(device)

#hyper parameters
PAD = '<PAD>'
START = '<START>'
END = '<END>'
UNK = '<UNK>'

vocab_size = 30000
max_sentence_length = 60

learning_rate = 1e-1
max_epochs = 800
min_epochs = 100
batch_size = 13

embedding_size = 512

patience = 10

resize_size = int(299/224*256)
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

#setup data stuff
print('reading data files...')
base_path_images = './data/flickr8k/Flicker8k_Dataset/'
reference_file = './data/flickr8k/Flickr8k_references.dev.json'
train_images = None
dev_images = None
test_images = None
annotations = None
with open('./data/flickr8k/Flickr_8k.trainImages.txt') as f:
    train_images = f.read().splitlines()
with open('./data/flickr8k/Flickr_8k.devImages.txt') as f:
    dev_images = f.read().splitlines()
with open('./data/flickr8k/Flickr_8k.testImages.txt') as f:
    test_images = f.read().splitlines()
with open('./data/flickr8k/Flickr8k.token.txt') as f:
    annotations = f.read().splitlines()

print('create/open vocabulary')
dev_processor = DataProcessor(annotations, dev_images, filename='./dev_flickr8k_vocab_'+str(vocab_size)+'.pkl', vocab_size=vocab_size)
dev_processor.save()
processor = DataProcessor(annotations, train_images, filename='./train_flickr8k_vocab_'+str(vocab_size)+'.pkl',vocab_size=vocab_size)
processor.save()

print('create data processing objects...')
traindata = data(base_path_images, train_images, annotations, max_sentence_length, processor, './data_flickr8k_train.pkl', START, END)
devdata = data(base_path_images, dev_images, annotations, max_sentence_length, processor, './data_flickr8k_dev.pkl', START, END)
testdata = data(base_path_images, test_images, annotations, max_sentence_length, processor, './data_flickr8k_test.pkl', START, END)

#create the models
print('create model...')
caption_model = CaptionModel(embedding_size, processor.vocab_size, device).to(device)
caption_model.train(True) #probably not needed. better to be safe
opt = SGD(caption_model.parameters(), lr=learning_rate)

losses = []
scores = []
best_bleu = 0
number_up = 0
opt.zero_grad()

#loop over number of epochs
print('training...')
for epoch in range(max_epochs):
    print('epoch %d'%epoch)
    #loop over all the training batches
    for i_batch, batch in enumerate(batch_generator(traindata,batch_size,transform_train, device)):
        image, caption, caption_lengths,_ = batch
        image = image.to(device)
        caption = caption.to(device)
        caption_lengths = caption_lengths.to(device)
        loss = caption_model(image, caption, caption_lengths)
        loss.backward()
        losses.append(float(loss))
        opt.step()

    #create validation result file
    print('validation...')
    caption_model.eval()
    encoder = caption_model.encoder.cuda()
    decoder = caption_model.decoder.cuda()

    predicted_sentences = dict()
    for image,image_name in batch_generator_dev(devdata,1,transform_eval, device):
        # Encode
        h0 = encoder(image)

        #prepare decoder initial hidden state
        h0 = h0.unsqueeze(0)
        c0 = torch.zeros(h0.shape).cuda()
        hidden_state = (h0,c0)

        # Decode
        start_token = torch.LongTensor([processor.w2i[START]]).to(device)
        predicted_words = []
        prediction = start_token.view(1,1)
        for w_idx in range(max_sentence_length):
            prediction, hidden_state = decoder(prediction, hidden_state)

            index_predicted_word = np.argmax(prediction.detach().cpu().numpy(), axis=2)[0][0]
            predicted_word = processor.i2w[index_predicted_word]
            predicted_words.append(predicted_word)

            if predicted_word == END:
                break
            prediction = torch.cuda.LongTensor([index_predicted_word]).view(1,1)
        predicted_sentences[image_name[0]] = predicted_words

        del(start_token)
        del(prediction)

    #perform validation
    prediction_file = 'output/dev_epoch_{}_baseline.pred'.format(epoch)
    with open(prediction_file, 'w', encoding='utf-8') as f:
        for im, p in predicted_sentences.items():
            if p[-1] == END:
                p = p[:-1]
            f.write(im+'\t'+' '.join(p) + '\n')

    score = evaluate(prediction_file, reference_file)
    scores.append(score)
    torch.save(caption_model.state_dict(), './output/last_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch))
    if len(scores) >= 1:
        if scores[-1]['Bleu_4'] <= best_bleu:
            number_up += 1
            if number_up > patience and epoch > min_epochs:
                print('Finished training!')
                break
        else:
            torch.save(caption_model.state_dict(), './output/best_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch))
            best_bleu = scores[-1]['Bleu_4']

    caption_model.train()

pickle.dump(scores, open('./output/scores_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))
pickle.dump(losses, open('./output/losses_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))

torch.save(caption_model.state_dict(), last_model_file_name)
