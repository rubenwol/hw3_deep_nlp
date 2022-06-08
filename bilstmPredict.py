import torch
from bilstmTrain import BiLSTMTagger
import sys
from tqdm import tqdm
from utils import *
import copy
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
BATCH_SIZE = 64

def prediction(dataloader, out_file, model, i2t):

    f = open(out_file, 'w')
    with torch.no_grad():
        for batch, (X, X_index, sentences) in enumerate(dataloader):
            # if batch <= 1397:
            #     continue
            x_word_index = None
            if model.rep == 'c':
                X = (X[0].to(device), X[1].to(device), X[2].to(device))
            elif model.rep == 'd':
                X = (X[0].to(device), X[1].to(device))
            else:
                X = X.to(device)
            if model.rep == 'b' or model.rep == 'd':
                X_index, x_word_index = X_index[0], X_index[1]

            probs = model(X, x_word_index)
            for prob, target_ind, sentence in zip(probs, X_index, sentences):
                pred = torch.argmax(prob[:target_ind + 1], dim=-1)
                for word, tag_idx in zip(sentence, pred):
                    f.write(f'{word} {i2t[tag_idx.item()]}\n')
                f.write("\n")
    f.close()






if __name__ == '__main__':
    # rep = sys.argv[1]
    # modelFile = sys.argv[2]
    # inputFile = sys.argv[3]
    # mapFile = sys.argv[4]
    rep = 'd'
    modelFile = 'model_ner'
    inputFile = 'ner/test'
    mapFile = 'mapFile'
    # LOAD MODEL
    kwargs, state = torch.load(modelFile)
    # kwargs['rep'] = rep
    model = BiLSTMTagger(**kwargs)
    model.load_state_dict(state)
    model = model.to(device)

    # Load mapFile
    dict = torch.load(mapFile)
    w2i = dict['w2i']
    c2i = dict['c2i']
    t2i = dict['t2i']
    pref2i = dict['pref2i']
    suf2i = dict['suf2i']

    if rep == 'a':
        sentences_raw_dev = read_test_data(inputFile)
        sentences_dev = rare_to_unk_test(copy.deepcopy(sentences_raw_dev), w2i)
        sentences_dev_id = sent_to_index(sentences_dev, w2i)
        dev_set = MyDataset(sentences_dev_id, sentences_raw_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pading_test)

    if rep == 'b':
        sentences_raw_dev = read_test_data(inputFile)
        sentences_dev_id, tags_dev_id, sentences_dev_char_id_pad, len_word_dev = sent_char_to_index(copy.deepcopy(sentences_raw_dev), c2i)
        dev_set = MyDataset(sentences_dev_char_id_pad, sentences_raw_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_padding_char_test)

    if rep == 'c':
        sentences_raw_dev, prefixes_dev, suffixes_dev = read_test_sub_data(inputFile)
        sentences_dev = rare_to_unk_test(copy.deepcopy(sentences_raw_dev), w2i)
        prefixes_dev = rare_to_unk_test(prefixes_dev, pref2i)
        suffixes_dev = rare_to_unk_test(suffixes_dev, suf2i)
        sentences_id_dev, prefixes_id_dev, suffixes_id_dev = sent_to_index(sentences_dev, w2i), sent_to_index(prefixes_dev, pref2i), sent_to_index(suffixes_dev, suf2i)
        dev_set = create_dataset_subword(sentences_id_dev, prefixes_id_dev, suffixes_id_dev, is_test=True, sentences=sentences_raw_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_padding_sub_test)

    if rep == 'd':
        sentences_raw_dev = read_test_data(inputFile)
        sentences_char_dev, _ = sent_to_sent_char(copy.deepcopy(sentences_raw_dev))
        sentences_dev = rare_to_unk_test(copy.deepcopy(sentences_raw_dev), w2i)

        sentences_id_dev = sent_to_index(sentences_dev, w2i)
        sentences_char_id, sentences_char_id_pad_dev, len_word_dev = sent_char_to_index(sentences_char_dev, c2i)
        dev_set = create_dataset_cat_word_char(sentences_id_dev, sentences_char_id_pad_dev, is_test=True, sentences=sentences_raw_dev)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_cat_word_char_test)

    i2t = {i: tag for tag, i in t2i.items()}
    prediction(dev_dataloader, 'test.ner', model, i2t)









