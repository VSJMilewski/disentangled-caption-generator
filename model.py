import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True).to(device)
        self.inception.aux_logits = False
        # the cnn is pretrained, so turn of the gradient
        for param in self.inception.parameters():
            param.requires_grad = False
        # replace the final fully connected layer to have the embedding size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.inception.fc.requires_grad = True
        self.init_weights()

    def forward(self, x):
        out = self.inception(x)
        return out

    def init_weights(self):
        nn.init.xavier_normal_(self.inception.fc.weight)
        nn.init.constant_(self.inception.fc.bias, 0.0)


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, hidden_size, embedding_size, lstm_layers=2, p=0.5):
        super().__init__()

        self.dropout = nn.Dropout(p=p)
        self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers, dropout=p)
        self.logit_lin = nn.Linear(hidden_size, target_vocab_size)  # out
        self.init_weights()

    def forward(self, input_words, hidden_input):
        # find the embedding of the correct word to be predicted
        emb = self.target_embeddings(input_words)
        emb = self.dropout(emb)
        # reshape to the correct order for the LSTM
        emb = torch.transpose(emb, 0, 1)
        # Put through the next LSTM step
        self.LSTM.flatten_parameters()
        lstm_output, hidden = self.LSTM(emb, hidden_input)
        output = self.logit_lin(lstm_output)
        output = output.transpose(0, 1)
        return output, hidden

    def init_weights(self):
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
        super().__init__()
        self.device = device
        self.lstm_layers = lstm_layers
        self.encoder = EncoderCNN(embedding_size, device).to(device)
        self.decoder = Decoder(target_vocab_size, hidden_size, embedding_size, lstm_layers=lstm_layers).to(device)
        self.h0_lin = nn.Linear(embedding_size, hidden_size)
        self.c0_lin = nn.Linear(embedding_size, hidden_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)
        self.init_weights()

    def forward(self, images, captions, caption_lengths):
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
        # _, hidden_state = self.decoder.LSTM(img_emb, hidden_state)  # start lstm with img emb at t=-1
        prediction, hidden_state = self.decoder(captions[:, :-1], hidden_state)
        out = self.loss(prediction.view(-1, prediction.shape[2]), captions[:, 1:].contiguous().view(-1))
        # normalize loss where each sentence is a different length
        out = torch.mean(torch.div(out.view(prediction.shape[0], prediction.shape[1]).sum(dim=1), caption_lengths))
        return out

    def init_weights(self):
        nn.init.xavier_normal_(self.h0_lin.weight)
        nn.init.xavier_normal_(self.c0_lin.weight)
        nn.init.constant_(self.h0_lin.bias, 0.0)
        nn.init.constant_(self.c0_lin.bias, 0.0)
#
# class BinaryDecoder(nn.Module):
#
#     def __init__(self, target_vocab_size, hidden_size, embedding_size, lstm_layers=2, p=0.5):
#         super().__init__()
#
#         self.dropout = nn.Dropout(p=p)
#         self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
#         self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers, dropout=p)
#         self.logit_lin = nn.Linear(embedding_size, target_vocab_size)  # out
#         self.init_weights()
#
#     def forward(self, input_words, hidden_input, topic_input, switch, z0):
#         # find the embedding of the correct word to be predicted
#         emb = self.target_embeddings(input_words)
#         emb = self.dropout(emb)
#         # reshape to the correct order for the LSTM
#         emb = torch.transpose(emb, 0, 1)
#
#         # Put through the next LSTM step
#         self.LSTM.flatten_parameters()
#         lstm_output, hidden = self.LSTM(emb, hidden_input)
#         output = self.logit_lin(lstm_output)
#         output = output.transpose(0, 1)
#         return output, hidden
#
#     def init_weights(self):
#         for name, param in self.LSTM.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
#             elif 'weight' in name:
#                 nn.init.xavier_normal_(param)
#         nn.init.xavier_uniform_(self.target_embeddings.weight)
#         nn.init.xavier_normal_(self.logit_lin.weight)
#         nn.init.constant_(self.logit_lin.bias, 0.0)
#
#
# class BinaryCaptionModel(nn.Module):
#     def __init__(self, hidden_size, embedding_size, target_vocab_size, lstm_layers, device, number_of_topics=100):
#         super().__init__()
#         self.device = device
#         self.lstm_layers = lstm_layers
#         self.encoder = EncoderCNN(embedding_size, device).to(device)
#         self.decoder = BinaryDecoder(target_vocab_size, hidden_size, embedding_size, lstm_layers=lstm_layers).to(device)
#         self.h0_lin = nn.Linear(embedding_size, hidden_size)
#         self.c0_lin = nn.Linear(embedding_size, hidden_size)
#         self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)
#
#         # general topic modelling
#         self.sent_topic_lin1 = nn.Linear(embedding_size, hidden_size)
#         self.sent_topic_lin2 = nn.Linear(hidden_size, number_of_topics)
#         self.mixing_linear1 = nn.Linear(switch_size, switch_size)
#         self.mixing_linear2 = nn.Linear(switch_size, switch_size)
#         self.desc_linear1 = nn.Linear(number_of_topics * 2, number_of_topics * 2)
#         self.desc_linear2 = nn.Linear(number_of_topics * 2, target_vocab_size)
#
#         self.init_weights()
#
#     def forward(self, images, captions, caption_lengths):
#         # Encode
#         img_emb = self.encoder(images)
#
#         # prepare decoder initial hidden state
#         img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
#         h0 = self.h0_lin(img_emb)
#         h0 = h0.repeat(self.lstm_layers, 1, 1)
#         c0 = self.c0_lin(img_emb)
#         c0 = c0.repeat(self.lstm_layers, 1, 1)
#         hidden_state = (h0, c0)
#
#         # predict sentence-level topic distribution
#         pi0 = F.relu(self.sent_topic_lin1(img_emb))
#         pi0 = F.relu(self.sent_topic_lin2(pi0))
#         pi0 = F.softmax(pi0)
#
#         # compute global topic embedding
#         z0 = F.mm
#
#         # Decode
#         # _, hidden_state = self.decoder.LSTM(img_emb, hidden_state)  # start lstm with img emb at t=-1
#         prediction, hidden_state = self.decoder(captions[:, :-1], hidden_state)
#         out = self.loss(prediction.view(-1, prediction.shape[2]), captions[:, 1:].contiguous().view(-1))
#         # normalize loss where each sentence is a different length
#         out = torch.mean(torch.div(out.view(prediction.shape[0], prediction.shape[1]).sum(dim=1), caption_lengths))
#         return out
#
#     def init_weights(self):
#         nn.init.xavier_normal_(self.h0_lin.weight)
#         nn.init.xavier_normal_(self.c0_lin.weight)
#         nn.init.xavier_normal_(self.sent_topic_lin1.weight)
#         nn.init.xavier_normal_(self.sent_topic_lin2.weight)
#         nn.init.constant_(self.h0_lin.bias, 0.0)
#         nn.init.constant_(self.c0_lin.bias, 0.0)
#         nn.init.constant_(self.sent_topic_lin1.bias, 0.0)
#         nn.init.constant_(self.sent_topic_lin2.bias, 0.0)
