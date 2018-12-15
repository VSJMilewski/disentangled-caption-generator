import numpy as np
import torch

from caption_eval.evaluations_function import evaluate


def validation_step(model, dataloader, processor, max_seq_length, pred_file, ref_file, criterion, device, beam_size=1):
    """
    Computes the loss (without backward propogation) for the given dataset and it computes the validation metrics
    :param model: The model on which to test
    :param dataloader: The dataloader used for geting batches
    :param processor: The dataprocessor with the vocabulary and index mappings
    :param max_seq_length: The maximum allowed length of to be predicted sequences
    :param pred_file: The file where the created predictions are written to
    :param ref_file: The json file with for each image references to their gold standard captions
    :param criterion: The defined criterion for computing the loss
    :param device: On which device to run the model
    :param beam_size: The beam size to use for making predictions. Beam_size == 1 will perform greedy search
    :return: The socres for the different metrics
             and the average loss over the entire dataset
    """
    val_loss = compute_validation_loss(model, dataloader, criterion, device)
    if beam_size == 1:
        predicted_sentences = greedy_validation(model, dataloader, processor, max_seq_length, device)
    else:
        predicted_sentences = beam_search_validation(model, dataloader, processor,
                                                     max_seq_length, device, beam_size=beam_size)

    # Compute score of metrics
    with open(pred_file, 'w', encoding='utf-8') as f:
        for im, p in predicted_sentences.items():
            if p:
                if p[-1] == processor.END:
                    p = p[:-1]
                f.write(im + '\t' + ' '.join(p) + '\n')
            else:
                f.write(im + '\t' + processor.UNK + '\n')
    score = evaluate(pred_file, ref_file)
    return score, val_loss


def compute_validation_loss(model, dataloader, criterion, device):
    """
    Computes the validation loss for the given dataset, no backprogation
    :param model: The model on which to test
    :param dataloader: The dataloader used for geting batches
    :param criterion: The defined criterion for computing the loss
    :param device: On which device to run the model
    :return: The average loss over the entire dataset
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            torch.cuda.synchronize()
            image, caption, _, caption_lengths = batch
            caption = caption[:, :int(caption_lengths.max().item())]
            image = image.to(device)
            caption = caption.to(device)
            caption_lengths = caption_lengths.to(device)
            prediction = model(image, caption)
            if type(prediction) is tuple:
                loss_lang = criterion(prediction[0].contiguous().view(-1, prediction[0].shape[2]),
                                      caption[:, 1:].contiguous().view(-1)).view(prediction[0].shape[0],
                                                                                 prediction[0].shape[1])
                loss_desc = criterion(prediction[1].contiguous().view(-1, prediction[1].shape[2]),
                                      caption[:, 1:].contiguous().view(-1)).view(prediction[1].shape[0],
                                                                                 prediction[1].shape[1])
                loss = prediction[2] * loss_lang + (1 - prediction[2]) * loss_desc
            else:
                loss = criterion(prediction.contiguous().view(-1, prediction.shape[2]),
                                 caption[:, 1:].contiguous().view(-1)).view(prediction.shape[0], prediction.shape[1])
            # compute the average loss over the average losses of each sample in the batch
            loss = torch.mean(torch.div(loss.sum(dim=1), caption_lengths)).sum()
            losses.append(float(loss))
    model.train()
    return np.mean(losses)


def greedy_validation(model, dataloader, processor, max_seq_length, device):
    """
    Creates predictings given the current state of the model using the greedy method, at each time step the
    word with the highest score is selected.
    :param model: The model on which to test
    :param dataloader: The dataloader used for geting batches
    :param processor: The dataprocessor with the vocabulary and index mappings
    :param max_seq_length: The maximum allowed length of to be predicted sequences
    :param device: On which device to run the model
    :return: The socres for the different metrics
    """
    model.eval()
    with torch.no_grad():
        predicted_sentences = dict()
        for image, _, image_names, _ in dataloader:
            torch.cuda.synchronize()
            image = image.to(device)
            input_ = torch.full((image.shape[0], 1), processor.w2i[processor.START], dtype=torch.long, device=device)
            if torch.cuda.device_count() > 1:
                predicted_ids = model.module.greedy_sample(image, input_, max_seq_length)
            else:
                predicted_ids = model.greedy_sample(image, input_, max_seq_length)
            # now derive the sentences
            predicted_ids = predicted_ids.cpu().data.numpy()
            for i, sentence_ids in enumerate(predicted_ids):
                caption = []
                for word_id in sentence_ids:
                    word = processor.i2w[word_id]
                    if word == processor.END:
                        break
                    caption.append(word)
                predicted_sentences[image_names[i]] = caption
    model.train()
    return predicted_sentences


def beam_search_validation(model, dataloader, processor, max_seq_length, device, beam_size=1):
    """
    Creates predictings given the current state of the model using the beam search method, at each time step the
    'beam_size' highest expansions on the current sequences are chosen. With these the beam is again expanded.
    :param model: The model on which to test
    :param dataloader: The dataloader used for geting batches
    :param processor: The dataprocessor with the vocabulary and index mappings
    :param max_seq_length: The maximum allowed length of to be predicted sequences
    :param device: On which device to run the model
    :param beam_size: The beam size to use for making predictions
    :return: The socres for the different metrics
    """
    model.eval()
    predicted_sentences = dict()
    with torch.no_grad():
        for image, _, image_names, _ in dataloader:
            image = image.to(device)
            if torch.cuda.device_count() > 1:
                predicted_sents = model.module.beam_sample(image, image_names, processor,
                                                           max_seq_length, beam_size)
            else:
                predicted_sents = model.beam_sample(image, image_names, processor, max_seq_length, beam_size)
            predicted_sentences.update(predicted_sents)
    model.train()
    return predicted_sentences
