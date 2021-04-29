from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
import json
import transformers
import torch

def compute_masks(mask):
    one_idx = [i for i, b in enumerate(mask) if b]
    zeros = torch.zeros(len(mask), dtype=torch.long)
    cls_mask, sep1_mask, sep2_mask = [torch.scatter(zeros, 0, torch.tensor(i), torch.tensor(1)) for i in one_idx]
    left_idx = torch.tensor(range(one_idx[0]+1, one_idx[1]))
    left_mask = torch.scatter(zeros, 0, left_idx, torch.ones(len(left_idx), dtype=torch.long))
    right_idx = torch.tensor(range(one_idx[1]+1, one_idx[2]))
    right_mask = torch.scatter(zeros, 0, right_idx, torch.ones(len(right_idx), dtype=torch.long))
    return {'cls_mask': cls_mask, 'sep1_mask': sep1_mask, 'sep2_mask': sep2_mask, 'left_mask': left_mask, 'right_mask': right_mask}

def coarse_lab(l):
    if len(l)>=2:
        if l[1] in "<>":
            return l[:2]
        else:
            return l[0]
    else:
        return l

def separate_flags(label, encoding_dict):
    flags = {'base': label[0],
             'direction': '<' if '<' in label else '>' if '>' in label else 'None',
             'has_i': 'i' in label,
             'has_s': 's' in label}
    return {k: encoding_dict[k][v] for k, v in flags.items()}

class PARADataset(torch.utils.data.Dataset):

    def __init__(self, data_list, bert_tokenizer, label_strategy, rew_dir=False, label_encoder=None):
        super().__init__()
        self.data_list = data_list
        self.bert_tokenizer = bert_tokenizer
        self.label_encoder = label_encoder
        self.flag_lab2i = {'base': {'2': 0, '3': 1, '4': 2},
                           'direction': {'None': 0, '<': 1, '>': 2},
                           'has_i': {False: 0, True: 1},
                           'has_s': {False: 0, True: 1}}

        if rew_dir:
            d = {'<': '>', '>': '<'}
            for i in range(len(self.data_list)):
                rew_label = ''.join(d[s] if s in d else s for s in self.data_list[i]['label'])
                self.data_list.append({'label': rew_label, 'txt1': self.data_list[i]['txt2'], 'txt2': self.data_list[i]['txt1']})

        if label_strategy == 'coarse':
            for entry in self.data_list:
                entry['label'] = coarse_lab(entry['label'])
        elif label_strategy == 'coarse-above-2':
            for entry in self.data_list:
                entry['label'] = coarse_lab(entry['label'])
            self.data_list = [d for d in self.data_list if d['label'][0] in '34']
        elif label_strategy == 'binary-4':
            self.flag_lab2i['base'] = {'1': 0, '4': 1}
            for entry in self.data_list:
                entry['label'] = '4' if entry['label'][0] == '4' else '1'
        elif label_strategy == 'binary-4-4s':
            self.flag_lab2i['base'] = {'1': 0, '4': 1}
            for entry in self.data_list:
                entry['label'] = '4' if entry['label'] == '4' or entry['label'] == '4s' else '1'
        elif label_strategy == 'with-1':
            self.flag_lab2i['base'] = {'1': 0, '2': 1, '3': 2, '4': 3}
        elif label_strategy == 'combined-neg':
            for entry in self.data_list:
                if entry['label'][0] == '1':
                    entry['label'] = '2'
        else: # If 1's are not included, leave them out of the data.
            self.data_list = [d for d in self.data_list if d['label'][0] != '1']

        if not self.label_encoder:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit([d['label'] for d in self.data_list])

        transformed = self.label_encoder.transform([d['label'] for d in self.data_list])
        for d, e in zip(self.data_list, transformed):
            d['label'] = e
    
    @classmethod
    def from_json(cls, fname, **args):
        data_list = []
        with open(fname, 'r') as f:
            for entry in json.load(f):
                data_list.append({k: entry[k] for k in ['label', 'txt1', 'txt2']})
                for txt_pair in entry['rewrites']:
                    data_list.append({'label': '4', 'txt1': txt_pair[0], 'txt2': txt_pair[1]})
        
        return cls(data_list, **args)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,key):
        item=self.data_list[key]
        t1_tok=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(item["txt1"]))
        t2_tok=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(item["txt2"]))
        encoded=self.bert_tokenizer.prepare_for_model(t1_tok, t2_tok, return_length=True, return_special_tokens_mask=True, max_length=512, truncation=True)
        original_label = self.label_encoder.inverse_transform([item['label']])[0]

        return {"input_ids":encoded.input_ids, "token_type_ids":encoded.token_type_ids, "attention_mask":encoded.attention_mask, "length":encoded.length, "label":item["label"], 'is_4': 1 if '4' in original_label else 0, **compute_masks(encoded.special_tokens_mask), **separate_flags(original_label, self.flag_lab2i)}

def collate(itemlist):
    batch={}
    for k in "input_ids","attention_mask","token_type_ids", "cls_mask", "sep1_mask", "sep2_mask", "left_mask", "right_mask":
        batch[k] = pad_with_zero([item[k] for item in itemlist])
    batch["label"] = torch.LongTensor([item["label"] for item in itemlist])
    batch['is_4'] = torch.LongTensor([item['is_4'] for item in itemlist])
    for k in 'base', 'direction', 'has_i', 'has_s':
        batch[k] = torch.LongTensor([item[k] for item in itemlist])
    return batch

def pad_with_zero(vals):
    vals=[torch.LongTensor(v) for v in vals]
    tokenized_single_batch=torch.nn.utils.rnn.pad_sequence(vals,batch_first=True)
    return tokenized_single_batch


class PARADataModule(pl.LightningDataModule):

    def __init__(self, batch_size, bert_model, label_strategy, train_fname, dev_fname, test_fname):
        super().__init__()
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.label_strategy = label_strategy
        self.train_fname = train_fname
        self.dev_fname = dev_fname
        self.test_fname = test_fname

    def setup(self,stage=None):
        self.bert_tokenizer=transformers.BertTokenizer.from_pretrained(self.bert_model)
        self.train_data=PARADataset.from_json(self.train_fname, bert_tokenizer=self.bert_tokenizer, label_strategy=self.label_strategy, rew_dir=True)
        self.dev_data=PARADataset.from_json(self.dev_fname, bert_tokenizer=self.bert_tokenizer, label_strategy=self.label_strategy, label_encoder=self.train_data.label_encoder)
        self.test_data=PARADataset.from_json(self.test_fname, bert_tokenizer=self.bert_tokenizer, label_strategy=self.label_strategy, label_encoder=self.train_data.label_encoder)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, collate_fn=collate, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_data, collate_fn=collate, batch_size=self.batch_size//2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, collate_fn=collate, batch_size=self.batch_size//2)


if __name__=="__main__":
    d=PARADataModule(".",100)
    d.setup()
    dl=d.train_dataloader()
    for x in dl:
        print("_")
    for x in dl:
        print("x")
