# Use a model trained on annotated para data to classify tatoeba data (or any other tsv files).

from para_data import PARADataset, collate
import csv
import time
import torch
import transformers

def transpose(l):
    return [list(t) for t in zip(*l)]

def line_batches(fn, line_batch_size):
    batch = []
    with open(fn) as f:
        for line in f:
            batch.append(line.rstrip('\n').split('\t'))
            if len(batch) >= line_batch_size:
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

def classify_tsv(model, bert_model, batch_size, fname, out_fname):
    line_batch_size = 100000
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_model)
    with open(out_fname, 'wt', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        start = time.time()
        for i, batch in enumerate(line_batches(fname, line_batch_size)):
            original_indexes, sorted_batch = transpose(sorted(enumerate(batch), key=lambda x: sum(len(bert_tokenizer.tokenize(t)) for t in x[1])))
            txt1, txt2 = transpose(sorted_batch)
            data_list = [{'label': '4', 'txt1': t1, 'txt2': t2} for t1, t2 in zip(txt1, txt2)]
            dataset = PARADataset(data_list=data_list, bert_tokenizer=bert_tokenizer, label_strategy=None, label_encoder=None)
            dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate, batch_size=batch_size)
            classified = classify(model, dataloader)
            if i == 0:
                header_keys = [k + ':' + str(h) for k in classified.keys() for h in dataset.flag_lab2i[k].keys()]
                tsv_writer.writerow([*header_keys, 'txt1', 'txt2'])
            classified = torch.cat([v for v in classified.values()], dim=1)
            
            for _, v, t1, t2 in sorted(zip(original_indexes, classified, txt1, txt2), key=lambda x: x[0]):
                tsv_writer.writerow([*v.tolist(), t1, t2])

            print(f"Batch {i+1} complete. Lines written: {i*line_batch_size + len(txt1)}. Batch time: {time.time() - start:.2f} s")
            start = time.time()
