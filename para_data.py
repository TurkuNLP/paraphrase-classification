import transformers
import torch
import pytorch_lightning as pl

class PARADataset(torch.utils.data.Dataset):

    def __init__(self,fname,bert_tokenizer,rew_dir=False):
        super().__init__()
        self.data_list=[]
        self.bert_tokenizer=bert_tokenizer
        self.lab2i={"3":0,"4":1,"4<":2,"4>":3} #TODO LOAD
        
        with open(fname,"rt") as f:
            for line in f:
                line=line.rstrip("\n")
                label,txt1,txt2=line.split("\t")
                self.data_list.append({"label":label,"txt1":txt1,"txt2":txt2})
                if rew_dir:
                    if ">" in label:
                        rew_label=label.replace(">","<")
                    elif "<" in label:
                        rew_label=label.replace("<",">")
                    else:
                        rew_label=label
                    self.data_list.append({"label":rew_label,"txt1":txt2,"txt2":txt1})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,key):
        item=self.data_list[key]
        t1_tok=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(item["txt1"]))
        t2_tok=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(item["txt2"]))
        encoded=self.bert_tokenizer.prepare_for_model(t1_tok,t2_tok,return_length=True) #todo overflow and whatnot
        
        return {"input_ids":encoded.input_ids,"token_type_ids":encoded.token_type_ids,"attention_mask":encoded.attention_mask,"length":encoded.length,"label":self.lab2i[item["label"]]}

def collate(itemlist):
    batch={}
    for k in "input_ids","attention_mask","token_type_ids":
        batch[k]=pad_with_zero([item[k] for item in itemlist])
    batch["label"]=torch.LongTensor([item["label"] for item in itemlist])
    return batch

def pad_with_zero(vals):
    vals=[torch.LongTensor(v) for v in vals]
    tokenized_single_batch=torch.nn.utils.rnn.pad_sequence(vals,batch_first=True)
    return tokenized_single_batch
    

class PARADataModule(pl.LightningDataModule):

    def __init__(self,data_dir,batch_size,bert_model="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__()
        self.batch_size=batch_size
        self.bert_model=bert_model

    def setup(self,stage=None):
        self.bert_tokenizer=transformers.BertTokenizer.from_pretrained(self.bert_model)
        self.train_data=PARADataset("para_train.tsv",self.bert_tokenizer,rew_dir=True)
        self.dev_data=PARADataset("para_dev.tsv",self.bert_tokenizer)
        self.test_data=PARADataset("para_test.tsv",self.bert_tokenizer)

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
