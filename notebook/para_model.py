import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch

class ParaMultiOutputAvgModel(pl.LightningModule):
    
    def __init__(self, bert_model, flag_lab2i, smooth=0, steps_train=None, weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.steps_train = steps_train
        self.weights = weights
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.flag_lab2i = flag_lab2i
        class_nums = {k: len(v) for k, v in self.flag_lab2i.items()}
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size*5, n) for name, n in class_nums.items()})
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.smooth = smooth


    def forward(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        cls_mask = batch['cls_mask']
        sep1_mask = batch['sep1_mask']
        sep2_mask = batch['sep2_mask']
        left_mask = batch['left_mask']
        right_mask = batch['right_mask']
        enc = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0] #BxS_LENxSIZE; BxSIZE
        cls = (enc*cls_mask.unsqueeze(-1)).sum(1) # enc.pooler_output
        sep1 = (enc*sep1_mask.unsqueeze(-1)).sum(1)
        sep2 = (enc*sep2_mask.unsqueeze(-1)).sum(1)
        left = (enc*left_mask.unsqueeze(-1)).sum(1) / left_mask.sum(-1).unsqueeze(-1)
        right = (enc*right_mask.unsqueeze(-1)).sum(1) / right_mask.sum(-1).unsqueeze(-1)
        catenated = torch.cat((cls, sep1, sep2, left, right), -1)

        return {name: layer(catenated) for name, layer in self.cls_layers.items()}
