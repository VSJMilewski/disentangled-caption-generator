import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from flickr8k_data_processor import batch_generator_dev
from caption_eval.evaluations_function import evaluate
from beam_search import Beam


def greedy_validation(model, data_set, processor, max_seq_length, transform, device, start='<start>', end='<end>'):
    model.eval()
    with torch.no_grad():
        enc = model.encoder.to(device)
        dec = model.decoder.to(device)

        predicted_sentences = dict()
        for image, image_name in batch_generator_dev(data_set, 1, transform, device):
            # Encode
            img_emb = enc(image)

            # expand the tensors to be of beam-size
            img_emb = img_emb.unsqueeze(0)
            c0 = torch.zeros(img_emb.shape).to(device)
            hidden_state = (img_emb, c0)

            # Decode
            _, hidden_state = dec.LSTM(img_emb, hidden_state)  # for t-1 put the imgage emb through the LSTM
            start_token = torch.LongTensor([processor.w2i[start]]).to(device)
            predicted_words = []
            prediction = start_token.view(1, 1)
            for w_idx in range(max_seq_length):
                prediction, hidden_state = dec(prediction, hidden_state)
                index_predicted_word = np.argmax(prediction.detach().cpu().numpy(), axis=2)[0][0]
                predicted_word = processor.i2w[index_predicted_word]
                predicted_words.append(predicted_word)

                if predicted_word == end:
                    break
                prediction = torch.LongTensor([index_predicted_word]).view(1, 1).to(device)
            predicted_sentences[image_name[0]] = predicted_words
            del start_token
            del prediction
    model.train()
    return predicted_sentences


def beam_search_validation(model, data_set, processor, max_seq_length, transform, device,
                           pad='<pad>', start='<start>', end='<end>', batch_size=1, beam_size=1):
    model.eval()
    with torch.no_grad():
        enc = model.encoder.to(device)
        dec = model.decoder.to(device)

        predicted_sentences = dict()
        for image, image_name in batch_generator_dev(data_set, batch_size, transform, device):
            # Encode
            img_emb = enc(image)

            # expand the tensors to be of beam-size
            img_emb = img_emb.unsqueeze(0)
            img_emb = img_emb.repeat(1, beam_size, 1)
            c0 = torch.zeros(img_emb.shape).to(device)
            hidden_state = (img_emb, c0)

            b_size = image.shape[0]

            # create the initial beam
            beam = [Beam(beam_size, processor.w2i, pad=pad, start=start, end=end, device=device)
                    for _ in range(b_size)]

            batch_idx = list(range(b_size))  # indicating index for every sample in the batch
            remaining_sents = b_size  # number of samples in batch

            # Decode
            _, hidden_state = dec.LSTM(img_emb, hidden_state)  # for t-1 put the imgage emb through the LSTM
            for w_idx in range(max_seq_length):
                input_ = torch.stack([b.get_current_state() for b in beam if not b.done]).view(-1, 1)
                out, hidden_state = dec(input_, hidden_state)
                out = F.softmax(out, dim=2)

                # process lstm step in beam search
                word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()
                active = []  # list of not finisched samples
                for b in range(b_size):
                    if beam[b].done:
                        continue

                    idx = batch_idx[b]
                    if not beam[b].advance(word_lk.data[idx]):  # returns true if complete
                        active.append(b)

                    for dec_state in hidden_state:  # iterate over h, c
                        sent_states = dec_state.view(-1, beam_size, remaining_sents, dec_state.size(2))[:, :, idx]
                        sent_states.data.copy_(sent_states.data.index_select(1, beam[b].get_current_origin()))

                # test if the beam is finished
                if not active:
                    break

                # in this section, the sentences that are still active are
                # compacted so that the decoder is not run on completed sentences
                active_idx = torch.LongTensor([batch_idx[k] for k in active]).to(device)
                batch_idx = {beam: idx for idx, beam in enumerate(active)}

                def update_active(t):
                    # select only the remaining active sentences
                    view = t.data.view(-1, remaining_sents, dec.hidden_size)
                    new_size = list(t.size())
                    new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                    return Variable(view.index_select(1, active_idx).view(*new_size))

                hidden_state = (update_active(hidden_state[0]), update_active(hidden_state[1]))
                remaining_sents = len(active)

            # select the best hypothesis
            for b in range(b_size):
                score_, k = beam[b].get_best()
                hyp = beam[b].get_hyp(k)
                predicted_sentences[image_name[b]] = [processor.i2w[idx.item()] for idx in hyp]

    model.train()
    return predicted_sentences


def compute_validation_loss(model, data_set, batch_size, transform, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i_batch, batch in enumerate(batch_generator_dev(data_set, batch_size, transform, device)):
            image, caption, caption_lengths, _ = batch
            image = image.to(device)
            caption = caption.to(device)
            caption_lengths = caption_lengths.to(device)
            loss = model(image, caption, caption_lengths)
            losses.append(float(loss))
    model.train()
    return np.mean(losses)


def validation_step(model, data_set, processor, max_seq_length, pred_file, ref_file, transform, device,
                    pad='<pad>', start='<start>', end='<end>', batch_size=1, beam_size=1):
    val_loss = compute_validation_loss(model, data_set, beam_size, transform, device)
    if beam_size == 1:
        predicted_sentences = greedy_validation(model, data_set, processor, max_seq_length, transform, device,
                                                start=start, end=end)
    else:
        predicted_sentences = beam_search_validation(model, data_set, processor, max_seq_length, transform, device,
                                                     pad=pad, start=start, end=end,
                                                     batch_size=batch_size, beam_size=beam_size)

    # Compute score of metrics
    with open(pred_file, 'w', encoding='utf-8') as f:
        for im, p in predicted_sentences.items():
            if p[-1] == end:
                p = p[:-1]
            f.write(im + '\t' + ' '.join(p) + '\n')
    score = evaluate(pred_file, ref_file)
    return score, val_loss
