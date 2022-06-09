import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from time import time
import numpy as np
from random import shuffle
from gen_examples import *
import sys
import string


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
EPOCHS = 200
BATCH_SIZE = 64 #64
EMB_DIM = LSTM_OUTPUT_DIM = 30
HIDDEN_LAYER = 50
print(f'BATCH SIZE : {BATCH_SIZE}')

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=BATCH_SIZE, bidirectional=False):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.D = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)


    def forward(self, x):
        # x = x.view(BATCH_SIZE, x.shape[0], x.shape[1])
        h_i = torch.zeros((x.size(0), self.hidden_size), device=device)
        c_i = torch.zeros((x.size(0), self.hidden_size), device=device)
        torch.nn.init.xavier_normal_(h_i)
        torch.nn.init.xavier_normal_(c_i)
        output = []
        for i in range(x.shape[1]):
            (h_i, c_i) = self.lstm_cell(x[:, i], (h_i, c_i))
            output.append(h_i)
        output = torch.stack(output, dim=1)
        return output, (h_i, c_i)

        # h_i = torch.zeros(self.hidden_size)
        # c_i = torch.zeros(self.hidden_size)
        # output = []
        # for i in range(len(x)):
        #     (h_i, c_i) = self.lstm_cell(x[i], (h_i, c_i))
        #     output.append(h_i)
        # output = torch.stack(output, dim=0)
        # return output, (h_i, c_i)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1)


class RNNAcceptor(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_lstm_dim, hidden_mlp, output_size):
        super(RNNAcceptor, self).__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_lstm_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_lstm_dim, batch_first=True)
        self.mlp = MLP(hidden_lstm_dim, hidden_mlp, output_size)

    def forward(self, x, x_index):
        emb = self.embeddings(x)
        out_lstm, (hn, cn) = self.lstm(emb)
        hn_without_pad = out_lstm[torch.arange(out_lstm.size(0)), x_index]
        return self.mlp(hn_without_pad)


def train(train_dataloader, model, loss_fn, optimizer):
    model.train()
    cum_loss = 0
    for (X, y, X_index) in tqdm(train_dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        probs = model(X, X_index)
        loss = loss_fn(probs, y)
        cum_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_train, loss_train = accuracy_and_loss(train_dataloader, model, loss_fn)
    acc_dev, loss_dev = accuracy_and_loss(test_dataloader, model, loss_fn)
    print(f"epoch: {epoch+1}, accuracy train: {acc_train}, loss train: {loss_train}, accuracy dev: {acc_dev}, loss train: {loss_dev} ")
    return acc_dev


def accuracy_and_loss(dataloader, model, loss_fn):
    model.eval()
    good = 0.0
    total = 0.0
    cum_loss = 0.0
    with torch.no_grad():
        for batch, (X, y, X_index) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            probs = model(X, X_index)
            loss = loss_fn(probs, y)
            cum_loss += loss
            pred = torch.argmax(probs.detach(), dim=-1)
            good += torch.sum((pred == y))
            total += len(y)

    return good / total, cum_loss / len(dataloader)

def collate_pading(batch):
    (x, y) = list(zip(*batch))
    x_pad = pad_sequence(x, batch_first=True, padding_value=len(char2ix))
    x_index = [len(x_i)-1 for x_i in x]
    y_pad = torch.LongTensor(y)
    return x_pad, y_pad, x_index


if __name__ == '__main__':
    lang = str(sys.argv[1])

    if lang == 'order':
        print('challenge is : ORDER')
        vocab = '123456789abcd'
        # pos_examples = np.loadtxt('pos_examples', dtype=str)
        # neg_examples = np.loadtxt('neg_examples', dtype=str)
        pos_examples = generate_pos_dataset(500)
        neg_examples = generate_neg_dataset(500)

    elif lang == 'palindrome':
        print('challenge is : PALINDROME')
        vocab = string.ascii_letters + string.digits
        pos_examples = generate_palindrome_dataset(1000)
        neg_examples = generate_random_seq_dataset(1000)

    elif lang == 'double':
        print('challenge is : DOUBLE WW')
        vocab = string.ascii_letters + string.digits
        pos_examples = generate_ww_dataset(1000)
        neg_examples = generate_random_seq_dataset(1000)

    elif lang == '0n1n':
        print(f'challenge is : \u03C3^n\u03BC^n for \u03C3 and \u03BC {chr(1013)} (a-z)')
        vocab = string.ascii_letters
        pos_examples = generate_0n_1n_dataset(500)
        neg_examples = generate_01_dataset(500)



    char2ix = {char: ix for ix, char in enumerate(vocab)}
    ix2char = {ix: char for char, ix in char2ix.items()}
    pos_examples = [[char2ix[c] for c in sample] for sample in pos_examples]
    pos_examples = [(torch.LongTensor(pos_sample), torch.LongTensor([1])) for pos_sample in pos_examples]

    neg_examples = [[char2ix[c] for c in sample] for sample in neg_examples]
    neg_examples = [(torch.LongTensor(neg_sample), torch.LongTensor([0])) for neg_sample in neg_examples]

    data = pos_examples + neg_examples
    shuffle(data)
    train_len = int(len(data) * 0.8)
    train_set = data[:train_len]
    dev_set = data[train_len:]

    # training_data, training_target, test_data, test_target = create_dataset()
    train_sample, train_target = zip(*train_set)
    dev_sample, dev_target = zip(*dev_set)
    train_set = MyDataset(train_sample, train_target)
    dev_set = MyDataset(dev_sample, dev_target)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pading)
    test_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pading)
    start_time = time()
    model = RNNAcceptor(vocab_size=len(vocab), embedding_dim=EMB_DIM, hidden_lstm_dim=LSTM_OUTPUT_DIM, hidden_mlp=HIDDEN_LAYER, output_size=2)
    model = model.to(device)
    loss_fn = nn.NLLLoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())


    for epoch in range(EPOCHS):
        accuracy_dev = train(train_dataloader, model, loss_fn, optimizer)
        if accuracy_dev == 1.0:
            print(f'SUCCES WITH {epoch+1} epochs')
            break
    end_time = time()
    print(f'program take {end_time - start_time} seconds')














