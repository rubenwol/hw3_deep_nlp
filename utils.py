RARE = 1
UNK = '<UNK>'
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MyTestDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def read_train_sub_data(fname):
    """
    fname: file name of the tagging data.
    structure of the file: each line contains a word SPACE tag pair. Blank lines are sentence boundaries
    return :sentences, vocab
    where : sentences = [sent0, sent1, ....,sentn]
            tags_sentences = [tags_sent0, tags_sent1, ... tags_sentn]
            senti = [word0, word1, ... wordk]
    """
    sentences, tags, sent, tags_sent, prefix_sent, suffix_sent, prefixes, suffixes = [], [], [], [], [], [], [], []
    vocab_words, vocab_prefix, vocab_suffix, vocab_tags = {}, {}, {}, {}
    f = open(fname, 'r')
    for line in f:
        if line == '\n':
            sentences.append(sent)
            prefixes.append(prefix_sent)
            suffixes.append(suffix_sent)
            tags.append(tags_sent)
            sent, tags_sent, prefix_sent, suffix_sent = [], [], [], []
        else:
            word, tag = line.split()
            word = word
            prefix = word[:3].lower()
            suffix = word[-3:].lower()
            sent.append(word)
            tags_sent.append(tag)
            prefix_sent.append(prefix)
            suffix_sent.append(suffix)
            vocab_words[word] = vocab_words.get(word, 0) + 1
            vocab_tags[tag] = vocab_words.get(word, 0) + 1
            vocab_prefix[prefix] = vocab_prefix.get(prefix, 0) + 1
            vocab_suffix[suffix] = vocab_prefix.get(suffix, 0) + 1
    f.close()
    return sentences, prefixes, suffixes, tags, vocab_words, vocab_tags, vocab_prefix, vocab_suffix

def read_train_data(fname):
    """
    fname: file name of the tagging data.
    structure of the file: each line contains a word SPACE tag pair. Blank lines are sentence boundaries
    return :sentences, vocab
    where : sentences = [sent0, sent1, ....,sentn]
            tags_sentences = [tags_sent0, tags_sent1, ... tags_sentn]
            senti = [word0, word1, ... wordk]
    """
    sentences, tags, sent, tags_sent = [], [], [], []
    vocab_words, vocab_tags = {}, {}

    with open(fname, 'r') as f:
        for line in f:
            if line == '\n':
                sentences.append(sent)
                tags.append(tags_sent)
                sent, tags_sent = [], []
            else:
                word, tag = line.split()
                sent.append(word.strip())
                tags_sent.append(tag)
                vocab_words[word] = vocab_words.get(word, 0) + 1
                vocab_tags[tag] = vocab_words.get(word, 0) + 1

    return sentences, tags, vocab_words, vocab_tags

def read_test_data(fname):
    """
    fname: file name of the tagging data.
    structure of the file: each line contains a word SPACE tag pair. Blank lines are sentence boundaries
    return :sentences, vocab
    where : sentences = [sent0, sent1, ....,sentn]
            tags_sentences = [tags_sent0, tags_sent1, ... tags_sentn]
            senti = [word0, word1, ... wordk]
    """
    sentences, sent = [], []
    with open(fname, 'r') as f:
        for line in f:
            if line == '\n':
                sentences.append(sent)
                sent = []
            else:
                word = line.strip()
                sent.append(word)
    return sentences

def read_test_sub_data(fname):
    """
    fname: file name of the tagging data.
    structure of the file: each line contains a word SPACE tag pair. Blank lines are sentence boundaries
    return :sentences, vocab
    where : sentences = [sent0, sent1, ....,sentn]
            tags_sentences = [tags_sent0, tags_sent1, ... tags_sentn]
            senti = [word0, word1, ... wordk]
    """
    sentences, sent, prefix_sent, suffix_sent, prefixes, suffixes = [], [], [], [], [], []
    f = open(fname, 'r')
    for line in f:
        if line == '\n':
            sentences.append(sent)
            prefixes.append(prefix_sent)
            suffixes.append(suffix_sent)
            sent, prefix_sent, suffix_sent = [], [], []
        else:
            word = line.strip()
            prefix = word[:3].lower()
            suffix = word[-3:].lower()
            sent.append(word)
            prefix_sent.append(prefix)
            suffix_sent.append(suffix)
    f.close()
    return sentences, prefixes, suffixes


def rare_to_unk(sentences, vocab_words):
    vocab_rare = set()
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if vocab_words.get(word, 0) <= RARE:
                sentence[i] = UNK
                vocab_rare.add(word)
    return sentences, vocab_rare

def rare_to_unk_test(sentences, w2i):
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in w2i:
                sentence[i] = UNK
    return sentences


def sent_to_index(sentences, w2i):
    sentences_id = [torch.LongTensor([w2i[word] for word in sentence]) for sentence in sentences]
    # tags_id = [torch.LongTensor([t2i[tag] for tag in tag_sent]) for tag_sent in tags]
    return sentences_id

def sent_char_to_index(sentences_char, c2i):
    sentences_id = [[[c2i.get(c, c2i[UNK]) for c in word] for word in sentence] for sentence in sentences_char]
    max_len_word = 0
    len_word = [[len(word) for word in sentence] for sentence in sentences_char]
    for l in len_word:
        max_len_word = max(l) if max(l) > max_len_word else max_len_word
    sentences_id_pad = []
    for sentence in sentences_id:
        sentence_id_pad = torch.zeros((len(sentence), max_len_word), dtype=torch.long)
        for i, word in enumerate(sentence):
            try:
                sentence_id_pad[i, :len(word)] = torch.LongTensor(word)
            except:
                print('BUG')
        sentences_id_pad.append(sentence_id_pad)
    # tags_id = [torch.LongTensor([t2i[tag] for tag in tag_sent]) for tag_sent in tags]
    return sentences_id, sentences_id_pad, len_word


def sent_to_sent_char(sentences):
    sentences_char = [[[c for c in word] for word in sentence] for sentence in sentences]
    vocab_char = dict()
    for sentence in sentences_char:
        for word in sentence:
            for c in word:
                vocab_char[c] = vocab_char.get(c, 0) + 1
    return sentences_char, vocab_char


def collate_pading_test(batch):
    (x, sentences) = list(zip(*batch))
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    x_index = [len(x_i)-1 for x_i in x]
    return x_pad, x_index, sentences

def collate_padding_char_test(batch):
    (x, sentences) = list(zip(*batch))
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    x_index = [len(x_i) - 1 for x_i in x]
    x_word_len = torch.LongTensor([[[x_pad_word.count_nonzero()-1] for x_pad_word in x_pad_sent] for x_pad_sent in x_pad]).squeeze()
    return x_pad, (x_index, x_word_len), sentences


def collate_padding_sub_test(batch):
    (x, sentences) = list(zip(*batch))
    words, pref, suf = list(zip(*x))
    words_pad = pad_sequence(words, batch_first=True, padding_value=0)
    pref_pad = pad_sequence(pref, batch_first=True, padding_value=0)
    suf_pad = pad_sequence(suf, batch_first=True, padding_value=0)
    x_index = [len(x_i) - 1 for x_i in words]
    x_pad = (words_pad, pref_pad, suf_pad)
    return x_pad, x_index, sentences

def collate_cat_word_char_test(batch):
    (x, sentences) = list(zip(*batch))
    words, char = list(zip(*x))
    words_pad = pad_sequence(words, batch_first=True, padding_value=0)
    char_pad = pad_sequence(char, batch_first=True, padding_value=0)
    char_word_len = torch.LongTensor([[[char_pad_word.count_nonzero()-1 if char_pad_word.count_nonzero()>0 else 0 ] for char_pad_word in x_pad_sent] for x_pad_sent in char_pad]).squeeze()
    x_index = [len(x_i) - 1 for x_i in words]
    x_pad = (words_pad, char_pad)
    return x_pad, (x_index, char_word_len), sentences

def create_dataset_subword(sentences_id, prefixes_id, suffixes_id, tags_id=None, is_test=False, sentences=None):
    data= []
    for sent, pref, suf in zip(sentences_id, prefixes_id, suffixes_id):
        data.append((sent, pref, suf))
    if is_test:
        dataset = MyDataset(data, sentences)
    else:
        dataset = MyDataset(data, tags_id)
    return dataset

def create_dataset_cat_word_char(sentences_id, sentences_char_id, tags_id=None, is_test=False, sentences=None):
    data = []
    for sent, char_sent in zip(sentences_id, sentences_char_id):
        data.append((sent, char_sent))
    if is_test:
        dataset = MyDataset(data, sentences)
    else:
        dataset = MyDataset(data, tags_id)
    return dataset
