# numpy for arrays/matrices/mathematical stuff
import numpy as np

# nltk for tokenizer
from nltk.tokenize import wordpunct_tokenize

# torch for the NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

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
from beam_search import Beam


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


def validation_step(model, data_set, processor, max_seq_length, pred_file, ref_file,
                    pad='<PAD>', start='<START>', end='<END>', beam_size=20, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        enc = model.encoder.to(device)
        dec = model.decoder.to(device)

        predicted_sentences = dict()
        for image, image_name in batch_generator_dev(data_set, 1, transform_eval, device):
            # Encode
            h0 = enc(image)

            # expand the tensors to be of beam-size
            h0 = h0.unsqueeze(0)
            h0 = h0.repeat(1, beam_size, 1)
            c0 = torch.zeros(h0.shape).to(device)
            hidden_state = (h0, c0)

            # create the initial beam
            beam = Beam(beam_size, processor.w2i, pad=pad, start=start, end=end, cuda=(device.type == 'cuda'))

            # Decode
            for w_idx in range(max_seq_length):
                input_ = beam.get_current_state().view(-1, 1)
                out, hidden_state = dec(input_, hidden_state)
                out = F.softmax(out, dim=2)

                # process lstm step in beam search
                active = not beam.advance(out.squeeze(0))  # returns true if complete
                for dec_state in hidden_state:  # iterate over h, c
                    dec_state.data.copy_(dec_state.data.index_select(1, beam.get_current_origin()))
                # test if the beam is finished
                if not active:
                    break

            # select the best hypothesis
            score_, k = beam.get_best()
            hyp = beam.get_hyp(k)
            predicted_sentences[image_name[0]] = [processor.i2w[idx.item()] for idx in hyp]

        # Compute score of metrics
    with open(pred_file, 'w', encoding='utf-8') as f:
        for im, p in predicted_sentences.items():
            if p[-1] == end:
                p = p[:-1]
            f.write(im + '\t' + ' '.join(p) + '\n')
    score = evaluate(pred_file, ref_file)
    model.train()
    return score


def print_info():
    print('=' * 80 + '\n{:8s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t{:7s}\n{}'.format(
        'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 80))


def print_score(score, time_, epoch):
    print('{:5d}   \t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:7.3f}s'.format(
        epoch, score['Bleu_1'] * 100, score['Bleu_2'] * 100, score['Bleu_3'] * 100, score['Bleu_4'] * 100,
               score['METEOR'] * 100, score['ROUGE_L'] * 100, score['CIDEr'] * 100, time_))


def train(config):
    # test if there is a gpu
    if config.device:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PAD = config.pad
    START = config.sos
    END = config.eos
    UNK = config.unk
    vocab_size = config.vocab_size
    max_seq_length = config.max_seq_length
    learning_rate = config.learning_rate
    max_epochs = config.max_epochs
    batch_size = config.batch_size  # 5 images per sample, 13x5=65, 25x5=125
    embedding_size = config.num_hidden
    patience = config.patience
    base_output_path = config.output_path
    base_data_path = config.data_path
    base_pickle_path = config.pickle_path

    # setup paths
    prediction_file = os.path.join(base_output_path, 'dev_baseline.pred')
    last_epoch_file = os.path.join(base_output_path, 'last_flickr8k_baseline_model.pkl')
    best_epoch_file = os.path.join(base_output_path, 'best_flickr8k_baseline_model.pkl')
    train_images_file = os.path.join(base_data_path, 'flickr8k/Flickr_8k.trainImages.txt')
    dev_images_file = os.path.join(base_data_path, 'flickr8k/Flickr_8k.devImages.txt')
    base_path_images = os.path.join(base_data_path, 'flickr8k/Flicker8k_Dataset/')
    reference_file = os.path.join(base_data_path, 'flickr8k/Flickr8k_references.dev.json')
    captions_file = os.path.join(base_data_path, 'flickr8k/Flickr8k.token.txt')
    train_vocab_file = os.path.join(base_pickle_path, 'train_flickr8k_vocab_{}.pkl'.format(vocab_size))
    train_data_file = os.path.join(base_pickle_path, 'data_flickr8k_train.pkl')
    dev_data_file = os.path.join(base_pickle_path, 'data_flickr8k_dev.pkl')

    # setup data stuff
    with open(train_images_file) as f:
        train_images = f.read().splitlines()
    with open(dev_images_file) as f:
        dev_images = f.read().splitlines()
    with open(captions_file) as f:
        annotations = f.read().splitlines()

    # data processor for vocab and their index mappings
    processor = DataProcessor(annotations, train_images, filename=train_vocab_file, vocab_size=vocab_size,
                              pad=PAD, start=START, end=END, unk=UNK)
    processor.save()

    # data files containing the data and handling batching
    train_data = data(base_path_images, train_images, annotations, max_seq_length, processor, train_data_file,
                      START, END)
    dev_data = data(base_path_images, dev_images, annotations, max_seq_length, processor, dev_data_file, START,
                    END)

    # create the models
    caption_model = CaptionModel(embedding_size, processor.vocab_size, device).to(device)
    params = list(caption_model.encoder.inception.fc.parameters()) + list(caption_model.decoder.parameters())
    opt = Adam(params, lr=learning_rate)

    # Start training
    losses = []
    avg_losses = dict()
    loss_current_ind = 0
    scores = []
    best_bleu = -1
    best_epoch = -1
    number_up = 0
    print('training started!')
    start0 = time.time()
    for epoch in range(max_epochs):
        start = time.time()

        # loop over all the training batches in the epoch
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
        # store epoch results
        avg_losses[len(losses)] = np.mean(losses[loss_current_ind:])
        loss_current_ind = len(losses)

        # validation
        score = validation_step(caption_model, dev_data, processor, max_seq_length,
                                prediction_file, reference_file, pad=PAD, start=START, end=END, device=device)
        scores.append(score)
        torch.save(caption_model.cpu().state_dict(), last_epoch_file)

        # test termination
        if score['Bleu_4'] <= best_bleu:
            number_up += 1
            if number_up > patience:
                break
        else:
            number_up = 0
            torch.save(caption_model.cpu().state_dict(), best_epoch_file)
            best_bleu = scores[-1]['Bleu_4']
            best_epoch = epoch

        # print some info
        end = time.time()
        if epoch % 50 == 0:
            print_info()
        print_score(scores[-1], end - start)

    print_info()
    print('\n\n\t\t --- Best Epoch: ---')
    print_info()
    print_score(scores[best_epoch], time.time() - start0)
    print('=' * 80)

    pickle.dump(scores, open('./output/scores_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))
    pickle.dump(losses, open('./output/losses_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))
    pickle.dump(avg_losses, open('./output/avg_losses_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch), 'wb'))

    torch.save(caption_model.cpu().state_dict(), './output/last_flickr8k_baseline_model_epoch_{}.pkl'.format(epoch))


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--sos', type=str, default='<START>', help='Default start of sentence token')
    parser.add_argument('--eos', type=str, default='<END>', help='Default end of sentence token')
    parser.add_argument('--pad', type=str, default='<PAD>', help='Default padding token')
    parser.add_argument('--unk', type=str, default='<UNK>', help='Default unknown token')
    parser.add_argument('--vocab_size', type=int, default=30000, help='Max size of the vocabulary')
    parser.add_argument('--patience', type=int, default=10, help='Patience before terminating')
    parser.add_argument('--max_seq_length', type=int, default=60, help='Length of an input sequence')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers in the model')
    parser.add_argument('--batch_size', type=int, default=25, help='Number of samples in batch, 5 sentences per sample')
    parser.add_argument('--dropout_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--output_path', type=str, default='output', help='location where to store the output')
    parser.add_argument('--data_path', type=str, default='data', help='location where to store data')
    parser.add_argument('--pickle_path', type=str, default='pickles', help='location where to store pickles')
    parser.add_argument('--device', type=str, default=None, help='On which device to run, cpu, cuda or None')

    config = parser.parse_args()

    # Train the model
    train(config)
