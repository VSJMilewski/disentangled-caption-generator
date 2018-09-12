import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True)
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


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.target_embeddings = nn.Embedding(target_vocab_size, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, embedding_size)
        self.logit_lin = nn.Linear(embedding_size, target_vocab_size)  # out

    def forward(self, input_words, hidden_input):
        # find the embedding of the correct word to be predicted
        emb = self.target_embeddings(input_words)
        # reshape to the correct order for the LSTM
        emb = emb.view(1, emb.size(0), self.embedding_size)
        # Put through the next LSTM step
        lstm_output, hidden = self.LSTM(emb, hidden_input)
        output = self.logit_lin(lstm_output)

        return output, hidden


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
        self.loss = nn.CrossEntropyLoss(ignore_index=0).to(device)

    def forward(self, images, captions, caption_lengths):
        # Encode
        h0 = self.encoder(images)
        print("h0")
        print(h0)
        # prepare decoder initial hidden state
        h0 = h0.unsqueeze(0)
        print(h0)
        print("\nc0")
        c0 = torch.zeros(h0.shape).to(self.device)
        hidden_state = (h0, c0)

        # Decode
        batch_size, max_sent_len = captions.shape
        out = torch.zeros((batch_size)).to(self.device)
        for w_idx in range(max_sent_len - 1):
            prediction, hidden_state = self.decoder(captions[:, w_idx].view(-1, 1), hidden_state)
            print("\nprediction")
            print(prediction)
            print("\nhidden_state")
            print(hidden_state)
            out += self.loss(prediction.squeeze(0), captions[:, w_idx + 1])
            print("out")
            print(out)
        # normalize loss
        out = torch.mean(torch.div(out,
                                   caption_lengths))  # the loss is the average of losses, so divide over number of words in each sentence

        return out
