# Use a model trained on annotated para data to calculate a similarity matrix from a list of sentences.

from para_data import PARADataset, collate
from math import sqrt
import numpy as np
import csv
import time
import torch
import transformers

def transpose(l):
    return [list(t) for t in zip(*l)]

def sentence_pair_batches(fn, batch_size):
    sentences = []
    # Slurp the file to produce all possible sentence pairs
    with open(fn) as f:
        # If a tsv file, the sentence should be first on each line
        sentences = [line.rstrip('\r\n').split('\t')[0] for line in f]
    
    batch = []
    for s1 in sentences:
        for s2 in sentences:
            batch.append((s1, s2))
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch

def classify(model, dataloader):
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            output = model({k: v.cuda() for k, v in batch.items()})
            preds.append(output)

    preds = [{k: torch.nn.functional.softmax(v, dim=-1) for k, v in b.items()} for b in preds]
    d = {}
    for k in preds[0].keys():
        d[k] = torch.cat([b[k] for b in preds], dim=0)
    return d

def cluster_tsv(model, bert_model, batch_size, label_strategy, fname, out_fname):
    line_batch_size = 100000
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_model)
    # bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_model, truncation=True, max_length=32)

    base4_list = []
    for batch in sentence_pair_batches(fname, line_batch_size):
        data_list = [{'label': '4', 'txt1': t1, 'txt2': t2} for t1, t2 in batch]
        dataset = PARADataset(data_list=data_list, bert_tokenizer=bert_tokenizer, label_strategy=label_strategy, label_encoder=None)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate, batch_size=batch_size)
        classified = classify(model, dataloader)
        index4 = list(dataset.flag_lab2i['base'].keys()).index('4')
        for base4 in classified['base'][:, index4]:
            base4_list.append(base4.item())

    matrix_len = int(sqrt(len(base4_list)))
    sim_matrix = np.array(base4_list).reshape(matrix_len, matrix_len)
    np.save(out_fname, sim_matrix)
    # ax = sns.heatmap(sim_matrix, vmin=0, vmax=1)
    # plt.savefig("heatmap.pdf")
