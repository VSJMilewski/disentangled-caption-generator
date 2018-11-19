import argparse
import os
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, RMSprop, Adagrad
from torch.utils.data import DataLoader
from torchvision import transforms

# import other files
from flickr_dataset import FlickrDataset
from model import CaptionModel, BinaryCaptionModel
from validation import validation_step
from vocab import DataProcessor


def print_info(file=None):
    """
    Prints the header information for printing the results.
    :param file: A file name where the information is printed to. If None, it prints to STDOUT
    :return: Nothing
    """
    if file:
        with open(file, 'a') as f:
            print('=' * 80 + '\n{:8s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t{:7s}\n{}'.format(
                'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 80), file=f)
    else:
        print('=' * 80 + '\n{:8s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t {:6s}\t{:7s}\n{}'.format(
            'EPOCH:', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'ROGUEl', 'CIDer', 'Time', '=' * 80))


def print_score(score, time_, epoch, file=None):
    """
    Prints the results of an epoch.
    :param score: A dict with containing scores for the different metrics: BLEU1-4, METEOR< ROUGE_L and CIDEr
    :param time_: The time in seconds the epoch took to run
    :param epoch: The epoch for which the results are printed
    :param file:  A file name where the information is printed to. If None, it prints to STDOUT
    :return: Nothing
    """
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
    """
    Print a final summary of the entire training process
    :param score: A dict with containing scores for the different metrics: BLEU1-4, METEOR< ROUGE_L and CIDEr
    :param time_: The time in seconds the epoch took to run
    :param epoch: The epoch for which the results are printed
    :param file:  A file name where the information is printed to. If None, it prints to STDOUT
    :return: Nothing
    """
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
    """
    The training process of the code
    :return: Nothing
    """

    # Load files containing the image files for the data divisions
    with open(train_images_file) as f:
        train_images = f.read().splitlines()
    with open(dev_images_file) as f:
        dev_images = f.read().splitlines()
    with open(test_images_file) as f:
        test_images = f.read().splitlines()
    with open(captions_file) as f:
        annotations = f.read().splitlines()

    # Setup a data processor for a vocabulary and index to word mappings.
    processor = DataProcessor(annotations, train_images, filename=train_vocab_file, vocab_size=config.vocab_size,
                              pad=config.pad, start=config.sos, end=config.eos, unk=config.unk,
                              vocab_threshold=config.vocab_threshold)

    # data files containing the data
    train_data = FlickrDataset(base_path_images, annotations, train_images, processor,
                               train_data_file, transform_train, config.max_seq_length, unique=False)
    dev_data = FlickrDataset(base_path_images, annotations, dev_images, processor,
                             dev_data_file, transform_eval, config.max_seq_length, unique=True)
    test_data = FlickrDataset(base_path_images, annotations, test_images, processor,
                              test_data_file, transform_eval, config.max_seq_length, unique=True)

    # create the dataloaders for handling batches
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers, drop_last=True)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,
                             pin_memory=True, num_workers=config.num_workers, drop_last=False)

    # create the chosen model
    model = None
    if config.model == 'BASELINE':
        model = CaptionModel(config.hidden_size, config.emb_size, processor.vocab_size,
                             config.lstm_layers, config.dropout_prob, device).to(device)
    elif config.model == 'BINARY':
        model = BinaryCaptionModel(config.hidden_size, config.emb_size, processor.vocab_size,
                                   config.lstm_layers, config.dropout_prob, device, config.binary_train_method,
                                   number_of_topics=config.number_of_topics).to(device)
    else:
        exit('not an existing model!')

    # if there are multiple GPUs, run the model in parallel on them
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # setup the chosen optimizer with all trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = None
    if config.optimizer == 'Adam':
        opt = Adam(params, lr=config.learning_rate)
    elif config.optimizer == 'RMSprop':
        opt = RMSprop(params, lr=config.learning_rate)
    elif config.optimizer == 'Adagrad':
        opt = Adagrad(params, lr=config.learning_rate)
    else:
        exit('None existing optimizer')

    # set the loss function. It has to ignore the padding and the reduction is done manually because of different
    # sequence lengths within a batch
    criterion = nn.CrossEntropyLoss(ignore_index=processor.w2i[config.pad], reduction='none').to(device)

    # Setup variables to keep track of training process
    losses = []
    avg_losses = dict()
    val_losses = dict()
    loss_current_ind = 0
    scores = []
    best_bleu = -1
    best_epoch = -1
    number_up = 0
    time_start0 = time.time()

    # start training
    for epoch in range(config.max_epochs):
        time_start = time.time()

        # loop over all the training batches in the epoch
        for i_batch, batch in enumerate(train_loader):
            # make sure the GPUs are synchronised at the start of the batch
            torch.cuda.synchronize()
            opt.zero_grad()
            image, caption, _, caption_lengths = batch
            # make sure the data is on the correct device
            image = image.to(device)
            caption = caption.to(device)
            caption_lengths = caption_lengths.to(device)

            # perform the forward pass through the model
            prediction = model(image, caption)
            # compute the loss by flattening all the results
            loss = criterion(prediction.contiguous().view(-1, prediction.shape[2]),
                             caption[:, 1:].contiguous().view(-1))
            # compute the average loss over the average losses of each sample in the batch
            loss = torch.mean(torch.div(loss.view(prediction.shape[0], prediction.shape[1]).sum(dim=1),
                                        caption_lengths)).sum()
            loss.backward()
            losses.append(float(loss))
            # clip the gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.max_grad)
            opt.step()
        # store epoch results
        avg_losses[len(losses)] = np.mean(losses[loss_current_ind:])
        loss_current_ind = len(losses)

        # validation
        score, val_loss = validation_step(model, dev_loader, processor, config.max_seq_length, prediction_file,
                                          reference_file, criterion, device, beam_size=config.beam_size)
        scores.append(score)
        val_losses[len(losses)] = val_loss
        # save the model and the results
        torch.save(model.cpu().state_dict(), last_epoch_file)
        model = model.to(device)
        pickle.dump(scores, open(score_pickle, 'wb'))
        pickle.dump(losses, open(loss_pickle, 'wb'))
        pickle.dump(avg_losses, open(avg_loss_pickle, 'wb'))
        pickle.dump(val_losses, open(val_loss_pickle, 'wb'))

        # test termination because of time
        time_end = time.time()
        if config.max_time and time_end - time_start0 > config.max_time:
            break

        # test if the model improved
        if score[config.eval_metric] <= best_bleu:
            number_up += 1
            # teset termination because we are out of patience
            if epoch > config.min_epochs and number_up > config.patience:
                break
        else:
            number_up = 0
            torch.save(model.cpu().state_dict(), best_epoch_file)
            model = model.to(device)
            best_bleu = scores[-1][config.eval_metric]
            best_epoch = epoch

        # print training progress
        if epoch % 50 == 0:
            print_info(progress_file)
        print_score(scores[-1], time_end - time_start, epoch, progress_file)

    print_final(scores[best_epoch], time.time() - time_start0, best_epoch, progress_file)
    model = model.cpu()
    model.load_state_dict(torch.load(best_epoch_file))
    model.to(device)
    # if there are multiple GPUs, run the model in parallel on them
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    test_score, _ = validation_step(model, test_loader, processor, config.max_seq_length, prediction_file,
                                    test_reference_file, criterion, device, beam_size=config.beam_size)

    if progress_file:
        with open(progress_file, 'a') as f:
            print('===== TEST =====', file=f)
    else:
        print('===== TEST =====')
    print_score(test_score, time.time() - time_start0, -1, progress_file)
    pickle.dump(test_score, open(test_score_pickle, 'wb'))


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
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of hidden units in the LSTM')
    parser.add_argument('--number_of_topics', type=int, default=100, help='For the topic modelling in the binary model')
    parser.add_argument('--emb_size', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers in the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of samples in batch')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers')
    parser.add_argument('--beam_size', type=int, default=20, help='size of the beam during eval, use 1 for greedy')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout keep probability')
    parser.add_argument('--min_epochs', type=int, default=0, help='Min number of training steps')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--output_path', type=str, default='output', help='location where to store the output')
    parser.add_argument('--data_path', type=str, default='data', help='location where to store data')
    parser.add_argument('--pickle_path', type=str, default='pickles', help='location where to store pickles')
    parser.add_argument('--model', type=str, default='BASELINE', help='which model to use: BASELINE, BINARY')
    parser.add_argument('--binary_train_method', type=str, default='WEIGHTED', help='how to train the switch: '
                                                                                    'WEIGHTED, SWITCHED')
    parser.add_argument('--dataset', type=str, default='flickr8k', help='flickr8k, flickr30k, coco(not ready yet)')
    parser.add_argument('--device', type=str, default='cuda', help='On which device to run, cpu, cuda or None')
    parser.add_argument('--num_workers', type=int, default=0, help='On how many devices to run, for more GPUs. '
                                                                   'For 4 GPUs, use 16' )
    parser.add_argument('--max_grad', type=float, default=5, help='max value for gradients')
    parser.add_argument('--vocab_threshold', type=int, default=5, help='minimum number of occurrences to be in vocab')
    parser.add_argument('--eval_metric', type=str, default='Bleu_4', help='on which metric to do early stopping')
    parser.add_argument('--unique', type=str, default='', help='string to make files unique')
    parser.add_argument('--progress_to_file', type=bool, default=True, help='if results should be printed to file')
    parser.add_argument('--max_time', type=int, default=None, help='fail save for job walltime')
    parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')

    config = parser.parse_args()
    # set which device to use
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

    pickle_unique = '{dataset}_{vocab}_th{threshold}'.format(dataset=config.dataset,
                                                              vocab=config.vocab_size,
                                                              threshold=config.vocab_threshold)
    file_unique = '{model}_{data}_beam{beam}_lstm{layers}_pat{pat}_emb{emb}_hidden{hidden}_p{p}_opt{opt}_grad{grad}'.format(
        model=config.model, data=config.dataset, beam=config.beam_size, layers=config.lstm_layers, pat=config.patience,
        emb=config.emb_size, hidden=config.hidden_size, p=config.dropout_prob, opt=config.optimizer,
        grad=config.max_grad)
    if config.model == 'BINARY':
        file_unique += '_topics{}'.format(config.number_of_topics)

    # create directories from the file uniques
    os.makedirs(os.path.join(config.output_path, file_unique), exist_ok=True)
    os.makedirs(os.path.join(config.pickle_path, pickle_unique), exist_ok=True)

    # temporary files
    prediction_file = os.path.join(config.output_path,file_unique, 'predictions.txt')
    progress_file = None
    if config.progress_to_file:
        progress_file = os.path.join(config.output_path, file_unique, ' progress.txt')

    # output files
    last_epoch_file = os.path.join(config.output_path, file_unique, 'last_epoch.pkl')
    best_epoch_file = os.path.join(config.output_path, file_unique, 'best_epoch.pkl')
    score_pickle = os.path.join(config.output_path, file_unique, 'scores_train.pkl')
    loss_pickle = os.path.join(config.output_path, file_unique, 'losses_train.pkl')
    avg_loss_pickle = os.path.join(config.output_path, file_unique, 'losses_train_avg.pkl')
    val_loss_pickle = os.path.join(config.output_path, file_unique, 'losses_eval.pkl')
    test_score_pickle = os.path.join(config.output_path, file_unique, 'scores_test.pkl')

    # pickle files
    train_vocab_file = os.path.join(config.pickle_path, pickle_unique, 'vocab_train.pkl')
    train_data_file = os.path.join(config.pickle_path, pickle_unique, 'data_train.pkl')
    dev_data_file = os.path.join(config.pickle_path, pickle_unique, 'data_eval.pkl')
    test_data_file = os.path.join(config.pickle_path, pickle_unique, 'data_test.pkl')

    # data files
    train_images_file = None
    dev_images_file = None
    test_images_file = None
    base_path_images = None
    reference_file = None
    test_reference_file = None
    captions_file = None
    if config.dataset == 'flickr8k':
        train_images_file = os.path.join(config.data_path, 'flickr8k/Flickr_8k.trainImages.txt')
        dev_images_file = os.path.join(config.data_path, 'flickr8k/Flickr_8k.devImages.txt')
        test_images_file = os.path.join(config.data_path, 'flickr8k/Flickr_8k.testImages.txt')
        base_path_images = os.path.join(config.data_path, 'flickr8k/Flicker8k_Dataset/')
        reference_file = os.path.join(config.data_path, 'flickr8k/Flickr8k_references.dev.json')
        test_reference_file = os.path.join(config.data_path, 'flickr8k/Flickr8k_references.test.json')
        captions_file = os.path.join(config.data_path, 'flickr8k/Flickr8k.token.txt')
    elif config.dataset == 'flickr30k':
        train_images_file = os.path.join(config.data_path, 'flickr30k/flickr30k.trainImages.txt')
        dev_images_file = os.path.join(config.data_path, 'flickr30k/flickr30k.devImages.txt')
        test_images_file = os.path.join(config.data_path, 'flickr30k/flickr30k.testImages.txt')
        base_path_images = os.path.join(config.data_path, 'flickr30k/flickr30k_images/')
        reference_file = os.path.join(config.data_path, 'flickr30k/flickr30k_references.dev.json')
        test_reference_file = os.path.join(config.data_path, 'flickr30k/flickr30k_references.test.json')
        captions_file = os.path.join(config.data_path, 'flickr30k/results_20130124.token')
    else:
        exit('Unknown dataset')

    # Train the model
    train()
