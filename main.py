import pickle
import os
import time
import argparse
import numpy as np
import torch
from torch.optim import Adam, RMSprop, Adagrad
from torchvision import transforms
# from torch.utils.data import DataLoader
# import pycocotools  # cocoAPI

# import other files
from model import CaptionModel  # , BinaryCaptionModel
from vocab_flickr8k import DataProcessor
from flickr8k_data_processor import batch_generator, data
from validation import validation_step


def print_info(file):
    if file:
        with open(file, 'a') as f:
            print('=' * 80 + '\n{:8s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t{:7s}\n{}'.format(
                'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 80), file=f)
    else:
        print('=' * 80 + '\n{:8s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t{:7s}\n{}'.format(
            'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 80))


def print_score(score, time_, epoch, file):
    if file:
        with open(file, 'a') as f:
            print('{:5d}   \t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:7.3f}s'.format(
                epoch, score['Bleu_1'] * 100, score['Bleu_2'] * 100, score['Bleu_3'] * 100, score['Bleu_4'] * 100,
                       score['METEOR'] * 100, score['ROUGE_L'] * 100, score['CIDEr'] * 100, time_), file=f)
    else:
        print('{:5d}   \t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:7.3f}s'.format(
            epoch, score['Bleu_1'] * 100, score['Bleu_2'] * 100, score['Bleu_3'] * 100, score['Bleu_4'] * 100,
                   score['METEOR'] * 100, score['ROUGE_L'] * 100, score['CIDEr'] * 100, time_))


def print_final(score, time_, epoch, file):
    print_info(file)
    if file:
        with open(file, 'a') as f:
            print('\n\n\t\t --- Best Epoch: ---', file=f)
    else:
        print('\n\n\t\t --- Best Epoch: ---')
    print_info(file)
    print_score(score, time_, epoch, file)
    if file:
        with open(file, 'a') as f:
            print('=' * 80, file=f)
    else:
        print('=' * 80)


def train():
    # setup data stuff
    with open(train_images_file) as f:
        train_images = f.read().splitlines()
    with open(dev_images_file) as f:
        dev_images = f.read().splitlines()
    with open(captions_file) as f:
        annotations = f.read().splitlines()

    # data processor for vocab and their index mappings
    processor = DataProcessor(annotations, train_images, filename=train_vocab_file, vocab_size=config.vocab_threshold,
                              pad=config.pad, start=config.sos, end=config.eos, unk=config.unk,
                              vocab_threshold=config.vocab_threshold)

    # data files containing the data and handling batching
    train_data = data(base_path_images, train_images, annotations, config.max_seq_length, processor, train_data_file,
                      config.sos,
                      config.eos)
    dev_data = data(base_path_images, dev_images, annotations, config.max_seq_length, processor, dev_data_file,
                    config.sos, config.eos)

    # create the models
    model = None
    if config.model == 'BASELINE':
        model = CaptionModel(config.num_hidden, processor.vocab_size, device).to(device)
    # elif config.model == 'BINARY':
    #     model = BinaryCaptionModel(config.num_hidden, processor.vocab_size, device).to(device)
    else:
        exit('not an existing model!')
    # params = list(model.encoder.inception.fc.parameters()) + list(model.decoder.parameters())
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = None
    if config.optimizer == 'Adam':
        opt = Adam(params, lr=config.learning_rate)
    elif config.optimizer == 'RMSprop':
        opt = RMSprop(params, lr=config.learning_rate)
    elif config.optimizer == 'Adagrad':
        opt = Adagrad(params, lr=config.learning_rate)

    # Start training
    losses = []
    avg_losses = dict()
    val_losses = dict()
    loss_current_ind = 0
    scores = []
    best_bleu = -1
    best_epoch = -1
    number_up = 0
    print('training started!')
    time_start0 = time.time()
    for epoch in range(config.max_epochs):
        time_start = time.time()

        # loop over all the training batches in the epoch
        for i_batch, batch in enumerate(batch_generator(train_data, config.batch_size, transform_train, device)):
            opt.zero_grad()
            image, caption, caption_lengths, _ = batch
            image = image.to(device)
            caption = caption.to(device)
            caption_lengths = caption_lengths.to(device)
            loss = model(image, caption, caption_lengths)
            loss.backward()
            losses.append(float(loss))
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.max_grad)
            opt.step()
        # store epoch results
        avg_losses[len(losses)] = np.mean(losses[loss_current_ind:])
        loss_current_ind = len(losses)

        # validation
        score, val_loss = validation_step(model, dev_data, processor, config.max_seq_length, prediction_file,
                                          reference_file, transform_eval, device,
                                          beam_size=config.beam_size, batch_size=config.eval_batch_size,
                                          pad=config.pad, start=config.sos, end=config.eos)
        scores.append(score)
        avg_losses[len(losses)] = val_loss
        torch.save(model.cpu().state_dict(), last_epoch_file)
        model = model.to(device)
        pickle.dump(scores, open(score_pickle, 'wb'))
        pickle.dump(losses, open(loss_pickle, 'wb'))
        pickle.dump(avg_losses, open(avg_loss_pickle, 'wb'))
        pickle.dump(val_losses, open(val_loss_pickle, 'wb'))

        # test termination
        time_end = time.time()
        if score[config.eval_metric] <= best_bleu:
            number_up += 1
            if (epoch > config.min_epochs and number_up > config.patience) \
                    or (config.max_time and time_end - time_start0 > config.max_time):
                break
        else:
            number_up = 0
            torch.save(model.cpu().state_dict(), best_epoch_file)
            model = model.to(device)
            best_bleu = scores[-1][config.eval_metric]
            best_epoch = epoch

        # print some info
        if epoch % 50 == 0:
            print_info(progress_file)
        print_score(scores[-1], time_end - time_start, epoch, progress_file)

    print_final(scores[best_epoch], time.time() - time_start0, best_epoch, progress_file)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--sos', type=str, default='<start>', help='Default start of sentence token')
    parser.add_argument('--eos', type=str, default='<end>', help='Default end of sentence token')
    parser.add_argument('--pad', type=str, default='<pad>', help='Default padding token')
    parser.add_argument('--unk', type=str, default='<unk>', help='Default unknown token')
    parser.add_argument('--vocab_size', type=int, default=30000, help='Max size of the vocabulary')
    parser.add_argument('--patience', type=int, default=10, help='Patience before terminating')
    parser.add_argument('--max_seq_length', type=int, default=100, help='Length of an input sequence')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers in the model')
    parser.add_argument('--batch_size', type=int, default=25, help='Number of samples in batch, 5 sentences per sample')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of samples in eval batch')
    parser.add_argument('--beam_size', type=int, default=20, help='size of the beam during eval, use 1 for greedy')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout keep probability')
    parser.add_argument('--min_epochs', type=int, default=0, help='Min number of training steps')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--output_path', type=str, default='output', help='location where to store the output')
    parser.add_argument('--data_path', type=str, default='data', help='location where to store data')
    parser.add_argument('--pickle_path', type=str, default='pickles', help='location where to store pickles')
    parser.add_argument('--model', type=str, default='BASELINE', help='which model to use: BASELINE, BINARY')
    parser.add_argument('--dataset', type=str, default='flickr8k', help='flickr8k, flickr30k, coco(not ready yet)')
    parser.add_argument('--device', type=str, default='cuda', help='On which device to run, cpu, cuda or None')
    parser.add_argument('--max_grad', type=float, default=5, help='max value for gradients')
    parser.add_argument('--vocab_threshold', type=int, default=5, help='minimum number of occurances to be in vocab')
    parser.add_argument('--eval_metric', type=str, default='Bleu_4', help='on which metric to do early stopping')
    parser.add_argument('--unique', type=str, default='', help='string to make files unique')
    parser.add_argument('--progress_to_file', type=bool, default=True, help='if results should be printed to file')
    parser.add_argument('--max_time', type=int, default=None, help='fail save for job walltime')
    parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')

    config = parser.parse_args()
    device = torch.device(config.device)

    # globals for data transformations
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

    # temporary files
    prediction_file = os.path.join(config.output_path,
                                   '{}_{}_beam{}_{}.pred'.format(
                                       config.model, config.dataset, config.beam_size, config.unique))
    progress_file = os.path.join(config.output_path,
                                 '{}_{}_beam{}_{}.out'.format(
                                     config.model, config.dataset, config.beam_size, config.unique))

    # output files
    last_epoch_file = os.path.join(config.output_path,
                                   'last_{}_beam{}_{}_{}.pkl'.format(
                                       config.dataset, config.beam_size, config.model, config.unique))
    best_epoch_file = os.path.join(config.output_path,
                                   'best_{}_beam{}_{}_{}.pkl'.format(
                                       config.dataset, config.beam_size, config.model, config.unique))
    score_pickle = os.path.join(config.output_path,
                                'scores_{}_{}_beam{}_{}.pkl'.format(
                                    config.model, config.dataset, config.beam_size, config.unique))
    loss_pickle = os.path.join(config.output_path,
                               'losses_{}_{}_beam{}_{}.pkl'.format(
                                   config.model, config.dataset, config.beam_size, config.unique))
    avg_loss_pickle = os.path.join(config.output_path,
                                   'avg_losses_{}_{}_beam{}_{}.pkl'.format(
                                       config.model, config.dataset, config.beam_size, config.unique))
    val_loss_pickle = os.path.join(config.output_path,
                                   'val_losses_{}_{}_beam{}_{}.pkl'.format(
                                       config.model, config.dataset, config.beam_size, config.unique))

    # pickle files
    train_vocab_file = os.path.join(config.pickle_path,
                                    'train_{}_vocab_{}_th_{}.pkl'.format(
                                        config.dataset, config.vocab_size, config.vocab_threshold))
    train_data_file = os.path.join(config.pickle_path,
                                   'data_{}_train_th_{}.pkl'.format(config.dataset, config.vocab_threshold))
    dev_data_file = os.path.join(config.pickle_path,
                                 'data_{}_dev_th_{}.pkl'.format(config.dataset, config.vocab_threshold))

    # data files
    train_images_file = None
    dev_images_file = None
    base_path_images = None
    reference_file = None
    captions_file = None
    if config.dataset == 'flickr8k':
        train_images_file = os.path.join(config.data_path, 'flickr8k/Flickr_8k.trainImages.txt')
        dev_images_file = os.path.join(config.data_path, 'flickr8k/Flickr_8k.devImages.txt')
        base_path_images = os.path.join(config.data_path, 'flickr8k/Flicker8k_Dataset/')
        reference_file = os.path.join(config.data_path, 'flickr8k/Flickr8k_references.dev.json')
        captions_file = os.path.join(config.data_path, 'flickr8k/Flickr8k.token.txt')
    elif config.dataset == 'flickr30k':
        train_images_file = os.path.join(config.data_path, 'flickr30k/flickr30k.trainImages.txt')
        dev_images_file = os.path.join(config.data_path, 'flickr30k/flickr30k.devImages.txt')
        base_path_images = os.path.join(config.data_path, 'flickr30k/flickr30k_images/')
        reference_file = os.path.join(config.data_path, 'flickr30k/flickr30k_references.dev.json')
        captions_file = os.path.join(config.data_path, 'flickr30k/results_20130124.token')
    else:
        exit('Unknown dataset')

    # Train the model
    train()
