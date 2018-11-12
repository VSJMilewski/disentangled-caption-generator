import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from beam_search import Beam


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size, device):
        """
        Initialize the encoder model, which is a pretrained inception CNN
        :param embedding_size: The size of the embedding, which is the output of the encoder
        :param device: The device on which the model is run
        """
        super().__init__()

        # load the pretrained inception model
        self.inception = models.inception_v3(pretrained=True).to(device)
        self.inception.aux_logits = False
        # Set the requires_grad to False to turn of training
        for param in self.inception.parameters():
            param.requires_grad = False
        # replace the final fully connected layer to create the needed output
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        # the new layer should be trained
        self.inception.fc.requires_grad = True
        # initialize the weights of the layer
        self.init_weights()

    def forward(self, x):
        """
        Pass the input through the encoder model
        :param x: The input images [batch_size, 3, 299, 299]
        :return: The created encoded embeddings [batch_size, embedding_size]
        """
        out = self.inception(x)
        return out

    def init_weights(self):
        """
        initialize the weights of the fully connected layer to be normal dirstibuted and the bias all zeros.
        :return: Nothing
        """
        nn.init.xavier_normal_(self.inception.fc.weight)
        nn.init.constant_(self.inception.fc.bias, 0.0)


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, hidden_size, embedding_size, lstm_layers=2, p=0.5):
        """
        Initialize the global baseline decoder model. This model generates captions given encoded images.
        :param target_vocab_size: The vocabulary size
        :param hidden_size: The size of the hidden layers
        :param embedding_size: The size of the embeddings
        :param lstm_layers: How many hidden layers in LSTM network
        :param p: The dropout probability
        """
        super().__init__()

        self.dropout = nn.Dropout(p=p)
        self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers, dropout=p)
        self.logit_lin = nn.Linear(hidden_size, target_vocab_size)
        self.init_weights()

    def forward(self, input_words, hidden_input):
        """

        :param input_words: A tensor with a batch of padded captions
        :param hidden_input: the initial input for the hidden layers
        :return:
        """
        # find the embedding of the correct word to be predicted
        emb = self.target_embeddings(input_words)
        # pass the embeddings through a dropout layer
        emb = self.dropout(emb)
        # LSTM requires sequence first and batch size second
        emb = torch.transpose(emb, 0, 1)
        # LSTM parameters require to be flattened, otherwise it can crash if run in parallel
        self.LSTM.flatten_parameters()
        # For training, put the entire sequence through at once
        lstm_output, hidden = self.LSTM(emb, hidden_input)
        # put the LSTM output through the linear layer to get the predictions
        output = self.logit_lin(lstm_output)
        # transpose the have the output batch first again
        output = output.transpose(0, 1)
        return output, hidden

    def init_weights(self):
        """
        Initialize the weights and biases of the layers. All the weights with a normal distribution and the bias
        with all zeros. The embeddings are initialized with a uniform distribution.
        :return: Nothing
        """
        for name, param in self.LSTM.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_normal_(self.logit_lin.weight)
        nn.init.constant_(self.logit_lin.bias, 0.0)


class CaptionModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, target_vocab_size, lstm_layers, device):
        """
        Initialize the captioning model. It combines both the encoder and decoder model.
        :param hidden_size: Size of the hidden layers
        :param embedding_size: Size of the embeddings
        :param target_vocab_size: Size of the vocabulary
        :param lstm_layers: The number of LSTM layers
        :param device: The device on which to run the model
        """
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.encoder = EncoderCNN(embedding_size, device).to(device)
        self.decoder = Decoder(target_vocab_size, hidden_size, embedding_size, lstm_layers=lstm_layers).to(device)
        self.h0_lin = nn.Linear(embedding_size, hidden_size)
        self.c0_lin = nn.Linear(embedding_size, hidden_size)
        self.init_weights()

    def forward(self, images, captions):
        """
        Processes the images through the encoder. From the output of the encoder, the hidden layer of the
        decoder is initialized. After this, the decoder is run given the sequence of captions.
        :param images: The images to translate [Batch_size, 3, 299, 299]
        :param captions: The captions to be predicted [Batch_size, sequence_length]
        :return: The predicted scores at each time step [Batch_size, sequence_length, vocabulary_size]
        """
        # Encode
        img_emb = self.encoder(images)
        # prepare decoder initial hidden state
        img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
        h0 = self.h0_lin(img_emb)
        h0 = h0.repeat(self.lstm_layers, 1, 1)
        c0 = self.c0_lin(img_emb)
        c0 = c0.repeat(self.lstm_layers, 1, 1)
        hidden_state = (h0, c0)
        # Decode
        prediction, hidden_state = self.decoder(captions[:, :-1], hidden_state)
        return prediction

    def greedy_sample(self, images, input_, max_seq_length):
        """
        Processes the images through the encoder. From the output of the encoder, the hidden layer of the
        decoder is initialized. After this, the decoder is used to predict captions given an initial input
        :param images: The images to translate [Batch_size, 3, 299, 299]
        :param input_: The initial input (usually start tokens) for the captions
        :param max_seq_length: the max length of the to be predicted captions
        :return: The predicted ids at each time step [Batch_size, sequence_length]
        """
        # Encode
        img_emb = self.encoder(images)
        # prepare decoder initial hidden state
        img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
        h0 = self.h0_lin(img_emb)
        h0 = h0.repeat(self.lstm_layers, 1, 1)
        c0 = self.c0_lin(img_emb)
        c0 = c0.repeat(self.lstm_layers, 1, 1)
        hidden_state = (h0, c0)
        # Decode
        predicted_ids = []
        for w_idx in range(max_seq_length):
            prediction, hidden_state = self.decoder(input_, hidden_state)
            input_ = torch.argmax(prediction, dim=2)
            predicted_ids.append(input_)

        # now derive the sentences
        predicted_ids = torch.cat(predicted_ids, 1)
        return predicted_ids

    def beam_sample(self, images, image_names, processor, max_seq_length, beam_size):
        """

        :param images:
        :param image_names:
        :param processor:
        :param max_seq_length:
        :param beam_size:
        :return:
        """
        predicted_sentences = dict()
        # Encode
        img_emb = self.encoder(images)
        # prepare decoder initial hidden state
        img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
        h0 = self.h0_lin(img_emb)
        h0 = h0.repeat(self.lstm_layers, beam_size, 1)
        c0 = self.c0_lin(img_emb)
        c0 = c0.repeat(self.lstm_layers, beam_size, 1)
        hidden_state = (h0, c0)

        b_size = images.shape[0]

        # create the initial beam
        beam = [Beam(beam_size, processor, device=self.device) for _ in range(b_size)]

        batch_idx = list(range(b_size))  # indicating index for every sample in the batch
        remaining_sents = b_size  # number of samples in batch

        # Decode
        for w_idx in range(max_seq_length):
            input_ = torch.stack([b.get_current_state() for b in beam if not b.done]).view(-1, 1)
            out, hidden_state = self.decoder(input_, hidden_state)
            out = F.softmax(out, dim=2)

            # process lstm step in beam search
            word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()
            active = []  # list of not finisched samples
            for b in range(b_size):
                # if the current sample is done, skip it
                if beam[b].done:
                    continue

                # get the original index of the sample
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
            active_idx = torch.LongTensor([batch_idx[k] for k in active]).to(self.device)
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t, hidden_size):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, hidden_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx).view(*new_size))

            hidden_state = (update_active(hidden_state[0], self.hidden_size),
                            update_active(hidden_state[1], self.hidden_size))
            remaining_sents = len(active)

        # select the best hypothesis
        for b in range(b_size):
            score_, k = beam[b].get_best()
            hyp = beam[b].get_hyp(k)
            predicted_sentences[image_names[b]] = [processor.i2w[idx.item()] for idx in hyp]
        return predicted_sentences

    def init_weights(self):
        """
        Initialises the weights and biases for the linear layers connecting the encoder and the decoder.
        :return: Nothing
        """
        nn.init.xavier_normal_(self.h0_lin.weight)
        nn.init.xavier_normal_(self.c0_lin.weight)
        nn.init.constant_(self.h0_lin.bias, 0.0)
        nn.init.constant_(self.c0_lin.bias, 0.0)


class BinaryCaptionModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, target_vocab_size, lstm_layers, device, number_of_topics=100):
        """
        Initialize the binary captioning model. It combines both the encoder and decoder model. It uses a switch to
        determine at each time step to use the language model or the description model as decoder.
        :param hidden_size: Size of the hidden layers
        :param embedding_size: Size of the embeddings
        :param target_vocab_size: Size of the vocabulary
        :param lstm_layers: The number of LSTM layers
        :param device: The device on which to run the model
        :param number_of_topics: The number of topics to use for the topic modelling
        """
        super().__init__()
        self.device = device
        self.lstm_layers = lstm_layers
        self.vocab_size = target_vocab_size
        self.encoder = EncoderCNN(embedding_size, device).to(device)
        self.decoder = Decoder(target_vocab_size, hidden_size, embedding_size, lstm_layers=lstm_layers).to(device)
        self.h0_lin = nn.Linear(embedding_size, hidden_size)
        self.c0_lin = nn.Linear(embedding_size, hidden_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)

        # layers for the binary switch and the description model
        self.topic_embeddings = nn.Embedding(number_of_topics, embedding_size)
        self.sent_topic_lin1 = nn.Linear(embedding_size, hidden_size)
        self.sent_topic_lin2 = nn.Linear(hidden_size, number_of_topics)
        self.switch_lin1 = nn.Linear(embedding_size * 2 + hidden_size, hidden_size)
        self.switch_lin2 = nn.Linear(hidden_size, hidden_size)
        self.pos_topic_lin1 = nn.Linear(embedding_size * 2 + hidden_size, hidden_size)
        self.pos_topic_lin2 = nn.Linear(hidden_size, number_of_topics)
        self.output_lin1 = nn.Linear(embedding_size * 2, hidden_size)
        self.output_lin2 = nn.Linear(hidden_size, target_vocab_size)
        self.init_weights()

    def forward(self, images, captions, caption_lengths):
        """

        :param images: The images to translate [Batch_size, 3, 299, 299]
        :param captions: The captions to be predicted [Batch_size, sequence_length]
        :param caption_lengths: The lengths of the samples in the batch
        :return: The predicted scores at each time step [Batch_size, sequence_length, vocabulary_size]
        """
        # Encode
        img_emb = self.encoder(images)

        # prepare decoder initial hidden state
        img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
        h0 = self.h0_lin(img_emb)
        h0 = h0.repeat(self.lstm_layers, 1, 1)
        c0 = self.c0_lin(img_emb)
        c0 = c0.repeat(self.lstm_layers, 1, 1)
        hidden_state = (h0, c0)

        # compute sentence mixing coefficient
        pi0 = F.relu(self.sent_topic_lin1(img_emb))
        pi0 = F.relu(self.sent_topic_lin2(pi0))
        pi0 = F.softmax(pi0)

        # compute global topic embedding
        z0 = F.matmul(pi0, self.topic_embeddings.weight)

        # loop over the sequence length
        predictions = torch.empty((captions.shape[0], captions.shape[1] - 1, self.vocab_size))
        for i in range(captions.shape[1] - 1):
            topic_features = torch.cat(img_emb, z0, hidden_state[0])

            # compute the switch
            Bi = torch.relu(self.switch_lin1(topic_features))
            Bi = torch.relu(self.switch_lin1(Bi))
            Bi = torch.sigmoid(Bi)

            # compute the language model output
            prediction_language_model, hidden_state = self.decoder(captions[:, i], hidden_state)

            # compute the description model output
            pii = F.relu(self.pos_topic_lin1(topic_features))
            pii = F.relu(self.pos_topic_lin2(pii))
            pii = F.softmax(pii)
            zi = F.matmul(pii, self.topic_embeddings.weight)
            output_features = torch.cat(z0, zi)
            out = F.relu(self.output_lin1(output_features))
            prediction_description_model = F.relu(self.output_lin2(out))

            mask = torch.round(Bi)
            predictions[mask, i, :] = prediction_language_model[mask, :]
            predictions[1 - mask, i, :] = prediction_description_model[1 - mask, :]
        return predictions

    def init_weights(self):
        """
        Initialises the weights and biases for the linear layers connecting the encoder and the decoder,
        The topic embeddings.
        :return: Nothing
        """
        nn.init.xavier_uniform_(self.topic_embeddings.weight)

        nn.init.xavier_normal_(self.h0_lin.weight)
        nn.init.xavier_normal_(self.c0_lin.weight)
        nn.init.constant_(self.h0_lin.bias, 0.0)
        nn.init.constant_(self.c0_lin.bias, 0.0)

        nn.init.xavier_normal_(self.sent_topic_lin1.weight)
        nn.init.xavier_normal_(self.sent_topic_lin2.weight)
        nn.init.constant_(self.sent_topic_lin1.bias, 0.0)
        nn.init.constant_(self.sent_topic_lin2.bias, 0.0)

        nn.init.xavier_normal_(self.switch_lin1.weight)
        nn.init.xavier_normal_(self.switch_lin2.weight)
        nn.init.constant_(self.switch_lin1.bias, 0.0)
        nn.init.constant_(self.switch_lin2.bias, 0.0)

        nn.init.xavier_normal_(self.pos_topic_lin1.weight)
        nn.init.xavier_normal_(self.pos_topic_lin2.weight)
        nn.init.constant_(self.pos_topic_lin1.bias, 0.0)
        nn.init.constant_(self.pos_topic_lin2.bias, 0.0)

        nn.init.xavier_normal_(self.output_lin1.weight)
        nn.init.xavier_normal_(self.output_lin2.weight)
        nn.init.constant_(self.output_lin1.bias, 0.0)
        nn.init.constant_(self.output_lin2.bias, 0.0)
