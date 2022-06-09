import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from utils import *
import json
import sys
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim.lr_scheduler as SC
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=50, bidirectional=False):
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


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=50):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_cell_forward = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.lstm_cell_backward = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        h_i_forward = torch.zeros((x.size(0), self.hidden_size), device=device)
        c_i_forward = torch.zeros((x.size(0), self.hidden_size), device=device)
        torch.nn.init.xavier_normal_(h_i_forward)
        torch.nn.init.xavier_normal_(c_i_forward)

        h_i_backward = torch.zeros((x.size(0), self.hidden_size), device=device)
        c_i_backward = torch.zeros((x.size(0), self.hidden_size), device=device)
        torch.nn.init.xavier_normal_(h_i_backward)
        torch.nn.init.xavier_normal_(c_i_backward)

        output_forward = []
        output_backward = []
        for i in range(x.shape[1]):
            (h_i_forward, c_i_forward) = self.lstm_cell_forward(x[:, i], (h_i_forward, c_i_forward))
            output_forward.append(h_i_forward)
            (h_i_backward, c_i_backward) = self.lstm_cell_forward(x[:, -(i+1)], (h_i_backward, c_i_backward))
            output_backward.append(h_i_forward)
        output_forward = torch.stack(output_forward, dim=1)
        output_backward = torch.stack(output_backward, dim=1).flip(dims=[0])
        output = torch.cat([output_forward, output_backward], axis=-1)
        h_i = torch.cat([h_i_forward, h_i_backward], axis=-1)
        c_i = torch.cat([c_i_forward, c_i_backward], axis=-1)
        return output, (h_i, c_i)



class BiLSTMTagger(nn.Module):

    def __init__(self,
                 rep,
                 vocab_size,
                 embedding_dim,
                 hidden_lstm_dim,
                 output_size,
                 embedding_char_dim=None,
                 p=0.5,
                 vocab_pref_size=None,
                 vocab_suf_size=None,
                 vocab_char_size=None
                 ):
        super(BiLSTMTagger, self).__init__()
        self.kwargs = {
                            'rep': rep,
                            'vocab_size': vocab_size,
                            'embedding_dim': embedding_dim,
                            'hidden_lstm_dim': hidden_lstm_dim,
                            'output_size': output_size,
                            'p': p,
                            'embedding_char_dim': embedding_char_dim,
                            'vocab_pref_size': vocab_pref_size,
                            'vocab_suf_size': vocab_suf_size,
                            'vocab_char_size': vocab_char_size
        }
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.rep = rep
        if rep == 'a' or rep == 'c':
            self.bilstm1 = BiLSTM(embedding_dim, hidden_lstm_dim)
        if rep == 'b':
            self.char_emb = nn.Embedding(vocab_char_size + 1, embedding_char_dim, padding_idx=0)
            self.bilstm1 = BiLSTM(embedding_char_dim, hidden_lstm_dim)
        if rep == 'c':
            self.pref_emb = nn.Embedding(vocab_pref_size + 1, embedding_dim, padding_idx=0)
            self.suf_emb = nn.Embedding(vocab_suf_size + 1, embedding_dim, padding_idx=0)
        if rep == 'd':
            self.char_emb = nn.Embedding(vocab_char_size + 1, embedding_char_dim, padding_idx=0)
            self.fc_word_char = nn.Linear(embedding_dim + embedding_char_dim, embedding_dim + embedding_char_dim)
            self.bilstm1 = BiLSTM(embedding_dim + embedding_char_dim, hidden_lstm_dim)
        self.lstm_char = LSTM(embedding_char_dim, embedding_char_dim)
        # self.bilstm1 = BiLSTM(embedding_dim, hidden_lstm_dim)
        self.bilstm2 = BiLSTM(hidden_lstm_dim * 2, hidden_lstm_dim)
        # self.bilstm1 = nn.LSTM(embedding_dim, hidden_lstm_dim, batch_first=True, bidirectional=True, num_layers=2)
        # self.bilstm2 = nn.LSTM(embedding_dim, hidden_lstm_dim, batch_first=True, bidirectional=True, num_layers=2)
        # self.lstm = nn.LSTM(embedding_dim, hidden_lstm_dim, batch_first=True)
        self.fc = nn.Linear(hidden_lstm_dim * 2, output_size)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, x_word_index=None):
        if self.rep == 'a':
            out = self.embeddings(x)
        if self.rep == 'b':
            emb = self.char_emb(x)
            emb = emb.view(-1, emb.size(2), emb.size(3))
            out_lstm, _ = self.lstm_char(emb)
            out = out_lstm[torch.arange(out_lstm.size(0)), x_word_index.view(-1)].view(x.size(0), x.size(1), -1)
        if self.rep == 'c':
            word, pref, suf = x
            word_emb = self.embeddings(word)
            pref_emb = self.pref_emb(pref)
            suf_emb = self.suf_emb(suf)
            out = word_emb + pref_emb + suf_emb
        if self.rep == 'd':
            x_word, x_char = x
            emb_char = self.char_emb(x_char)
            emb_char = emb_char.view(-1, emb_char.size(2), emb_char.size(3))
            out_lstm, _ = self.lstm_char(emb_char)
            out_char = out_lstm[torch.arange(out_lstm.size(0)), x_word_index.view(-1)].view(x_char.size(0), x_char.size(1), -1)
            emb_word = self.embeddings(x_word)
            out = torch.cat([emb_word, out_char], axis=-1)
            out = self.fc_word_char(out)

        out_lstm, (hn, cn) = self.bilstm1(out)
        out_lstm, (hn, cn) = self.bilstm2(out_lstm)
        # hn_without_pad = out_lstm[torch.arange(out_lstm.size(0)), x_index]
        out = self.dropout(out_lstm)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


def train(train_dataloader, model, loss_fn, optimizer):
    model.train()
    cum_loss = 0
    count = 0
    acc_devs = []
    for batch, (X, y, X_index, y_index) in enumerate(tqdm(train_dataloader)):
        # Compute prediction and loss
        count += y.size(0)
        loss = 0.0
        number = 0
        x_word_index = None
        if model.rep == 'c':
            X = (X[0].to(device), X[1].to(device), X[2].to(device))
        elif model.rep == 'd':
            X = (X[0].to(device), X[1].to(device))
        else:
            X = X.to(device)
        y = y.to(device)
        if model.rep == 'b' or model.rep == 'd':
            X_index, x_word_index = X_index[0], X_index[1]
        probs = model(X, x_word_index)
        for prob, target, target_ind in zip(probs, y, y_index):
            loss += loss_fn(prob[:target_ind+1], target[:target_ind+1])
            number += 1
        cum_loss += loss/number

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if count % 500 == 0:
            acc_dev, loss_dev = accuracy_and_loss(dev_dataloader, model, loss_fn)
            acc_devs.append(acc_dev.item())


    acc_train, loss_train = accuracy_and_loss(train_dataloader, model, loss_fn)
    acc_dev, loss_dev = accuracy_and_loss(dev_dataloader, model, loss_fn)
    acc_devs.append(acc_dev.item())
    print(f"epoch: {epoch+1}, accuracy train: {acc_train}, loss train: {loss_train}, accuracy dev: {acc_dev}, loss train: {loss_dev} ")
    return acc_devs


def accuracy_and_loss(dataloader, model, loss_fn):
    model.eval()
    good = 0.0
    total = 0.0
    cum_loss = 0.0
    with torch.no_grad():
        for batch, (X, y, X_index, y_index) in enumerate(dataloader):
            loss = 0.0
            number = 0
            x_word_index = None
            if model.rep == 'c':
                X = (X[0].to(device), X[1].to(device), X[2].to(device))
            elif model.rep == 'd':
                X = (X[0].to(device), X[1].to(device))
            else:
                X = X.to(device)
            y = y.to(device)
            if model.rep == 'b' or model.rep == 'd':
                X_index, x_word_index = X_index[0], X_index[1]
            probs = model(X, x_word_index)
            for prob, target, target_ind in zip(probs, y, y_index):
                number += 1
                loss += loss_fn(prob[:target_ind + 1], target[:target_ind + 1])
                pred = torch.argmax(prob[:target_ind + 1], dim=-1)
                good += torch.sum((pred == target[:target_ind + 1]))
                total += len(target[:target_ind + 1])
                if NER:
                    int_o = t2i['O']

                    good -= len(pred[(pred == int_o) & (target[:target_ind + 1] == int_o)])
                    total -= len(pred[(pred == int_o) & (target[:target_ind + 1] == int_o)])
            loss /= number

            cum_loss += loss

    return good / total, cum_loss / len(dataloader)

def collate_pading(batch):
    (x, y) = list(zip(*batch))
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    x_index = [len(x_i)-1 for x_i in x]
    y_pad = pad_sequence(y, batch_first=True, padding_value=len(t2i))
    y_last_index = [len(y_i)-1 for y_i in y]
    return x_pad, y_pad, x_index, y_last_index

def collate_padding_char(batch):
    (x, y) = list(zip(*batch))
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    x_index = [len(x_i) - 1 for x_i in x]
    x_word_len = torch.LongTensor([[[x_pad_word.count_nonzero()-1] for x_pad_word in x_pad_sent] for x_pad_sent in x_pad]).squeeze()
    y_pad = pad_sequence(y, batch_first=True, padding_value=len(t2i))
    y_last_index = [len(y_i) - 1 for y_i in y]
    return x_pad, y_pad, (x_index, x_word_len), y_last_index


def collate_padding_sub(batch):
    (x, y) = list(zip(*batch))
    words, pref, suf = list(zip(*x))
    words_pad = pad_sequence(words, batch_first=True, padding_value=0)
    pref_pad = pad_sequence(pref, batch_first=True, padding_value=0)
    suf_pad = pad_sequence(suf, batch_first=True, padding_value=0)
    y_pad = pad_sequence(y, batch_first=True, padding_value=len(t2i))
    x_index = [len(x_i) - 1 for x_i in words]
    y_last_index = [len(y_i) - 1 for y_i in y]
    x_pad = (words_pad, pref_pad, suf_pad)
    return x_pad, y_pad, x_index, y_last_index

def collate_cat_word_char(batch):
    (x, y) = list(zip(*batch))
    words, char = list(zip(*x))
    words_pad = pad_sequence(words, batch_first=True, padding_value=0)
    char_pad = pad_sequence(char, batch_first=True, padding_value=0)
    char_word_len = torch.LongTensor([[[char_pad_word.count_nonzero()-1] for char_pad_word in x_pad_sent] for x_pad_sent in char_pad]).squeeze()
    y_pad = pad_sequence(y, batch_first=True, padding_value=len(t2i))
    x_index = [len(x_i) - 1 for x_i in words]
    y_last_index = [len(y_i) - 1 for y_i in y]
    x_pad = (words_pad, char_pad)
    return x_pad, y_pad, (x_index, char_word_len), y_last_index


if __name__ == '__main__':
    rep = sys.argv[1]
    trainFile = sys.argv[2]
    modelFile = sys.argv[3]
    devFile = sys.argv[4]
    mapFile = sys.argv[5]
    NER = int(sys.argv[6])
    print(rep)

    if rep == 'a' and NER:
        args = {'EPOCHS': 5,
                'BATCH_SIZE': 50,
                'EMB_DIM': 200,
                'LSTM_OUTPUT_DIM': 100,
                'P': 0.5,
                'EMB_CHAR_DIM': 100,
                'LR': 0.003}

    if rep == 'a' and not NER:
        args = {
            'EPOCHS': 5,
            'BATCH_SIZE': 50,
            'EMB_DIM': 200,
            'LSTM_OUTPUT_DIM': 100,
            'P': 0.5,
            'EMB_CHAR_DIM': 100,
            'LR': 0.003
        }

    if rep == 'b' and NER:
        args = {
            'EPOCHS': 5,
            'BATCH_SIZE': 50,
            'EMB_DIM': 200,
            'LSTM_OUTPUT_DIM': 400,
            'P': 0.5,
            'EMB_CHAR_DIM': 100,
            'LR': 0.01
        }

    if rep == 'b' and not NER:
        args = {'EPOCHS': 5,
                'BATCH_SIZE': 50,
                'EMB_DIM': 200,
                'LSTM_OUTPUT_DIM': 400,
                'P': 0.5,
                'EMB_CHAR_DIM': 30,
                'LR': 0.01}

    if rep == 'c' and NER:
        args = {'EPOCHS': 5, 'BATCH_SIZE': 50, 'EMB_DIM': 200, 'LSTM_OUTPUT_DIM': 100, 'P': 0.5, 'EMB_CHAR_DIM': 30,
                'LR': 0.003}

    if rep == 'c' and not NER:
        args = {'EPOCHS': 5, 'BATCH_SIZE': 50, 'EMB_DIM': 200, 'LSTM_OUTPUT_DIM': 100, 'P': 0.5, 'EMB_CHAR_DIM': 100,
                'LR': 0.003}

    if rep == 'd' and NER:
        args = {'EPOCHS': 5, 'BATCH_SIZE': 50, 'EMB_DIM': 200, 'LSTM_OUTPUT_DIM': 200, 'P': 0.5, 'EMB_CHAR_DIM': 30,
                'LR': 0.002}

    if rep == 'd' and not NER:
        args = {'EPOCHS': 5, 'BATCH_SIZE': 50, 'EMB_DIM': 100, 'LSTM_OUTPUT_DIM': 200, 'P': 0.5, 'EMB_CHAR_DIM': 30,
                'LR': 0.002}

    EPOCHS = args['EPOCHS']
    BATCH_SIZE = args['BATCH_SIZE']
    EMB_DIM = args['EMB_DIM']
    LSTM_OUTPUT_DIM = args['LSTM_OUTPUT_DIM']
    P = args['P']
    EMB_CHAR_DIM = args['EMB_CHAR_DIM']
    LR = args['LR']

    vocab_pref_size = None
    vocab_suf_size = None
    vocab_char_size = None
    w2i = {}
    c2i = {}
    suf2i = {}
    pref2i = {}
    t2i = {}
    if rep == 'a':
        sentences, tags, vocab_words, vocab_tags = read_train_data(trainFile)
        sentences, vocab_rare = rare_to_unk(sentences, vocab_words)
        vocab_words[UNK] = 1
        vocab = {word for word in vocab_words if word not in vocab_rare}
        vocab_size = len(vocab)
        w2i = {word: i+1 for i, word in enumerate(vocab)}
        t2i = {tag: i for i, tag in enumerate(vocab_tags)}
        sentences_id, tags_id = sent_to_index(sentences, w2i), sent_to_index(tags, t2i)
        train_set = MyDataset(sentences_id, tags_id)
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pading)

        sentences_dev, tags_dev, _, _ = read_train_data(devFile)
        sentences_dev, _ = rare_to_unk(sentences_dev, vocab_words)
        sentences_dev_id, tags_dev_id = sent_to_index(sentences_dev, w2i), sent_to_index(tags_dev, t2i)
        dev_set = MyDataset(sentences_dev_id, tags_dev_id)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pading)

    if rep == 'b':
        sentences, tags, vocab_words, vocab_tags = read_train_data(trainFile)
        sentences_char, vocab_char = sent_to_sent_char(sentences)
        vocab_char[UNK] = 1
        c2i = {c: i+1 for i, c in enumerate(vocab_char) if vocab_char[c] >= RARE}
        t2i = {tag: i for i, tag in enumerate(vocab_tags)}
        sentences_char_id, sentences_char_id_pad, len_word = sent_char_to_index(sentences_char, c2i)
        tags_id = sent_to_index(tags, t2i)
        vocab_size = len(vocab_words)
        vocab_char_size = len(c2i) + 1
        train_set = MyDataset(sentences_char_id_pad, tags_id)
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_padding_char)

        sentences_dev, tags_dev, _, _ = read_train_data(devFile)
        sentences_dev_id, sentences_dev_char_id_pad, len_word_dev = sent_char_to_index(sentences_dev, c2i)
        tags_dev_id = sent_to_index(tags_dev, t2i)
        dev_set = MyDataset(sentences_dev_char_id_pad, tags_dev_id)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_padding_char)

    if rep == 'c':
        sentences, prefixes, suffixes, tags, vocab_words, vocab_tags, dic_prefix, dic_suffix = read_train_sub_data(trainFile)
        sentences, vocab_rare = rare_to_unk(sentences, vocab_words)
        prefixes, vocab_prefix_rare = rare_to_unk(prefixes, dic_prefix)
        suffixes, vocab_suffix_rare = rare_to_unk(suffixes, dic_suffix)
        vocab_words[UNK] = dic_prefix[UNK] = dic_suffix[UNK] = 1
        vocab = {word for word in vocab_words if word not in vocab_rare}
        vocab_size = len(vocab)
        vocab_prefix = {pref for pref in dic_prefix if pref not in vocab_prefix_rare}
        vocab_pref_size = len(vocab_prefix)
        vocab_suffix = {suf for suf in dic_suffix if suf not in vocab_suffix_rare}
        vocab_suf_size = len(vocab_suffix)
        w2i = {word: i+1 for i, word in enumerate(vocab)}
        t2i = {tag: i for i, tag in enumerate(vocab_tags)}
        pref2i = {pref: i+1 for i, pref in enumerate(vocab_prefix)}
        suf2i = {suf: i+1 for i, suf in enumerate(vocab_suffix)}
        sentences_id, prefixes_id, suffixes_id, tags_id = sent_to_index(sentences, w2i), sent_to_index(prefixes, pref2i), sent_to_index(suffixes, suf2i), sent_to_index(tags, t2i)
        train_set = create_dataset_subword(sentences_id, prefixes_id, suffixes_id, tags_id)
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_padding_sub)

        sentences_dev, prefixes_dev, suffixes_dev, tags_dev, _, _, _, _ = read_train_sub_data(devFile)
        sentences_dev, _ = rare_to_unk(sentences_dev, vocab_words)
        prefixes_dev, _ = rare_to_unk(prefixes_dev, dic_prefix)
        suffixes_dev, _ = rare_to_unk(suffixes_dev, dic_suffix)
        sentences_id_dev, prefixes_id_dev, suffixes_id_dev, tags_id_dev = sent_to_index(sentences_dev, w2i), sent_to_index(prefixes_dev, pref2i), sent_to_index(suffixes_dev, suf2i), sent_to_index(tags_dev, t2i)
        dev_set = create_dataset_subword(sentences_id_dev, prefixes_id_dev, suffixes_id_dev, tags_id_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_padding_sub)

    if rep == 'd':
        sentences, tags, vocab_words, vocab_tags = read_train_data(trainFile)
        sentences_char, vocab_char = sent_to_sent_char(sentences)

        vocab_size = len(vocab_words)
        vocab_char_size = len(vocab_char)

        sentences, vocab_rare = rare_to_unk(sentences, vocab_words)
        vocab_words[UNK], vocab_char[UNK] = 1, 1

        vocab = {word for word in vocab_words if word not in vocab_rare}
        vocab_size = len(vocab)

        w2i = {word: i + 1 for i, word in enumerate(vocab)}
        t2i = {tag: i for i, tag in enumerate(vocab_tags)}
        c2i = c2i = {c: i + 1 for i, c in enumerate(vocab_char) if vocab_char[c] >= RARE}
        vocab_char_size = len(c2i) + 1

        sentences_id, tags_id = sent_to_index(sentences, w2i), sent_to_index(tags, t2i)
        sentences_char_id, sentences_char_id_pad, len_word = sent_char_to_index(sentences_char, c2i)

        train_set = create_dataset_cat_word_char(sentences_id, sentences_char_id_pad, tags_id)
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_cat_word_char)

        sentences_dev, tags_dev, _, _ = read_train_data(devFile)
        sentences_char_dev, _ = sent_to_sent_char(sentences_dev)

        sentences_dev, _ = rare_to_unk(sentences_dev, vocab_words)

        sentences_id_dev, tags_id_dev = sent_to_index(sentences_dev, w2i), sent_to_index(tags_dev, t2i)
        sentences_char_id, sentences_char_id_pad_dev, len_word_dev = sent_char_to_index(sentences_char_dev, c2i)
        dev_set = create_dataset_cat_word_char(sentences_id_dev, sentences_char_id_pad_dev, tags_id_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_cat_word_char)

    # emb_dim_list = [100, 200]
    # emb_char_dim_list = [30, 100]
    # lstm_output_dim_list = [100, 400]

    # params = [(a, b, c) for a in emb_dim_list for b in emb_char_dim_list for c in lstm_output_dim_list]

    best_acc = 0
    # for param in params:
    #     EMB_DIM = args['EMB_DIM'] = param[0]
    #     EMB_CHAR_DIM = args['EMB_CHAR_DIM'] = param[1]
    #     LSTM_OUTPUT_DIM = args['LSTM_OUTPUT_DIM'] = param[2]
    start_time = time()
    model = BiLSTMTagger(rep=rep,
                         vocab_size=vocab_size,
                         embedding_dim=EMB_DIM,
                         hidden_lstm_dim=LSTM_OUTPUT_DIM,
                         output_size=len(vocab_tags),
                         embedding_char_dim=EMB_CHAR_DIM,
                         p=P,
                         vocab_pref_size=vocab_pref_size,
                         vocab_suf_size=vocab_suf_size,
                         vocab_char_size=vocab_char_size)

    model = model.to(device)
    loss_fn = nn.NLLLoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['LR'])
    print('*' * 40)
    print(args)
    # scheduler = SC.StepLR(optimizer, step_size=2, gamma=0.1)
    acc_devs_array = []
    for epoch in range(5):
        accuracy_devs = train(train_dataloader, model, loss_fn, optimizer)
        acc_devs_array.extend(accuracy_devs)
        # if accuracy_dev == 1.0:
        #     print(f'SUCCES WITH {epoch+1} epochs')
        #     break
        # scheduler.step()
    print(args)
    print('*' * 40)
    end_time = time()
    print(f'program take {end_time - start_time} seconds')
    task = 'ner' if NER else 'pos'
    with open(f'dev_acc/dev_acc_{task}_rep_{rep}_{args}.json', 'w') as f:
        json.dump(acc_devs_array, f)
    accuracy_dev, loss_dev = accuracy_and_loss(dev_dataloader, model, loss_fn)
    if accuracy_dev > best_acc:
        # torch.save(model.state_dict(), modelFile)
        torch.save([model.kwargs, model.state_dict()], modelFile)
        best_acc = accuracy_dev

    torch.save({
                'w2i': w2i,
                't2i': t2i,
                'c2i': c2i,
                'pref2i': pref2i,
                'suf2i': suf2i
    }, mapFile)














