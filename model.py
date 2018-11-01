import torch
import torch.nn as nn
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

    def forward(self, x):
        out = self.inception(x)
        return out

    def init_weights(self):
        nn.init.xavier_normal(self.inception.fc.weight)
        nn.init.constant(self.inception.fc.bias, 0.0)


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, p=0.5):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.dropout = nn.Dropout(p=p)
        self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, embedding_size)
        self.logit_lin = nn.Linear(embedding_size, target_vocab_size)  # out

    def forward(self, input_words, hidden_input):
        # find the embedding of the correct word to be predicted
        emb = self.target_embeddings(input_words)
        emb = self.dropout(emb)
        # reshape to the correct order for the LSTM
        emb = emb.view(emb.shape[1], emb.shape[0], self.embedding_size)
        # Put through the next LSTM step
        lstm_output, hidden = self.LSTM(emb, hidden_input)
        output = self.logit_lin(lstm_output)
        output = output.view(output.shape[1], output.shape[0], output.shape[2])
        return output, hidden

    def init_weights(self):
        for name, param in self.LSTM.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.target_embeddings.weight)
        nn.init.xavier_normal(self.logit_lin.weight)
        nn.init.constant(self.logit_lin.bias, 0.0)


class CaptionModel(nn.Module):
    def __init__(self,
                 embedding_size,
                 target_vocab_size,
                 device):
        super().__init__()
        self.device = device
        self.target_vocab_size = target_vocab_size
        self.encoder = EncoderCNN(embedding_size, device).to(device)
        self.decoder = Decoder(target_vocab_size, embedding_size).to(device)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)

    def forward(self, images, captions, caption_lengths):
        # Encode
        img_emb = self.encoder(images)
        # prepare decoder initial hidden state
        img_emb = img_emb.unsqueeze(0)  # seq len, batch size, emb size
        c0 = torch.zeros(img_emb.shape).to(self.device)
        hidden_state = (img_emb, c0)
        # Decode
        _, hidden_state = self.decoder.LSTM(img_emb, hidden_state)
        prediction, hidden_state = self.decoder(captions[:, :-1], hidden_state)
        out = self.loss(prediction.view(-1, prediction.shape[2]), captions[:, 1:].contiguous().view(-1))
        # normalize loss where each sentence is a different length
        out = torch.mean(torch.div(out.view(prediction.shape[0], prediction.shape[1]).sum(dim=1), caption_lengths))
        return out

# class BinaryDecoder(nn.Module):
#     def __init__(self, target_vocab_size, embedding_size, switch_size, number_of_topics, topic_emb):
#         super().__init__()
#
#         self.topic_emb = topic_emb
#         self.embedding_size = embedding_size
#         self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
#         self.LSTM = nn.LSTM(embedding_size, embedding_size)
#         # output layer
#         self.logit_lin = nn.Linear(embedding_size, target_vocab_size)
#
#         # general topic modelling
#         self.relu = nn.ReLU()
#         self.mixing_linear1 = nn.Linear(switch_size, switch_size)
#         self.mixing_linear2 = nn.Linear(switch_size, switch_size)
#         self.topic_linear1 = nn.Linear(switch_size, switch_size)
#         self.topic_linear2 = nn.Linear(switch_size, number_of_topics)
#         self.desc_linear1 = nn.Linear(number_of_topics * 2, number_of_topics * 2)
#         self.desc_linear2 = nn.Linear(number_of_topics * 2, target_vocab_size)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input_words, hidden_input, topic_input, switch, z0):
#         # find the embedding of the correct word to be predicted
#         emb = self.target_embeddings(input_words)
#         # reshape to the correct order for the LSTM
#         emb = emb.view(1, emb.size(0), self.embedding_size)
#
#         # Put through the next LSTM step
#         lstm_output, hidden = self.LSTM(emb, hidden_input)
#         output = self.logit_lin(lstm_output)
#
#         # Put through the description model
#         # Mixing coefficient
#         r31 = self.relu(self.mixing_linear1(topic_input))
#         r32 = self.relu(self.mixing_linear2(r31))
#         # topic modelling
#         r41 = self.relu(self.topic_linear1(r32))
#         r42 = self.relu(self.topic_linear2(r41))
#         zi = torch.matmul(r42, self.topic_emb)
#         desc_input = torch.cat((z0, zi), 1)
#         r51 = self.relu(self.topic_linear1(desc_input))
#         r52 = self.relu(self.topic_linear1(r51))
#         desc_output = self.softmax(r52)
#
#         # binary selection
#         output[switch] = desc_output[switch]
#         hidden[switch] = hidden_input[switch]
#
#         return output, hidden


# class BinaryCaptionModel(nn.Module):
#     def __init__(self,
#                  embedding_size,
#                  target_vocab_size,
#                  number_of_topics,
#                  device):
#         super().__init__()
#         self.device = device
#         self.target_vocab_size = target_vocab_size
#         self.number_of_topics = number_of_topics
#         switch_size = embedding_size * 2 + number_of_topics
#         self.topic_emb = nn.Embedding(number_of_topics, embedding_size)
#         self.relu = nn.ReLU()
#
#         self.encoder = EncoderCNN(embedding_size).to(device)
#         self.decoder = Decoder(target_vocab_size, embedding_size, switch_size, number_of_topics, self.topic_emb).to(
#             device)
#         self.loss = nn.CrossEntropyLoss(ignore_index=0).to(device)
#
#
#         # general topic modelling
#         self.topic_linear1 = nn.Linear(embedding_size, embedding_size)
#         self.topic_linear2 = nn.Linear(embedding_size, number_of_topics)
#
#         # binary switch
#         self.switch_linear1 = nn.Linear(switch_size, switch_size)
#         self.switch_linear2 = nn.Linear(switch_size, 2)
#
#     def forward(self, images, captions, caption_lengths):
#         # Encode
#         h0 = self.encoder(images)
#         # prepare decoder initial hidden state
#         h0 = h0.unsqueeze(0)
#         c0 = torch.zeros(h0.shape).to(self.device)
#         hidden_state = (h0, c0)
#
#         # general topics
#         r11 = self.relu(self.topic_linear1(h0))
#         r12 = self.relu(self.topic_linear2(r11))
#
#         embs = self.topic_emb(torch.arange(self.number_of_topics).to(device))
#         z0 = torch.matmul(r12, embs)
#
#         # Decode
#         batch_size, max_sent_len = captions.shape
#         out = torch.zeros((batch_size)).to(self.device)
#         for w_idx in range(max_sent_len - 1):
#             # binary switch
#             switch_input = torch.cat((h0, z0, hidden_state[0]), 1)
#             r21 = self.relu(self.switch_linear1(switch_input))
#             r22 = self.relu(self.switch_linear2(r21))
#             switch = torch.argmax(r22, 1)
#
#             prediction, hidden_state = self.decoder(captions[:, w_idx].view(-1, 1), hidden_state, switch, z0)
#
#
#             out += self.loss(prediction.squeeze(0), captions[:, w_idx + 1])
#         # normalize loss where each sentence is a different length
#         out = torch.mean(torch.div(out,
#                                    caption_lengths))
#
#         return out
