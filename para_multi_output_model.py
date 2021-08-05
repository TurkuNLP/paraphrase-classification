import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch

def model_output_to_p(out):
    return {k: v.argmax(-1) for k, v in out.items()}

# Encodes prediction and target 2D-vectors as 1D-vectors for use with the accuracy metric.
def encode_tensors(pred, target):
    base = max(pred.max(), target.max()).item() + 1
    return encode_tensor(pred, base), encode_tensor(target, base)

def encode_tensor(tensor, base):
    return torch.tensor([sum(base**i * n.item() for i, n in enumerate(v)) for v in tensor])

def smooth_loss(out, target, smooth, ignored=None):
    logp = F.log_softmax(out, dim=-1)
    nll_loss = -logp.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logp.mean(dim=-1)
    loss = (1-smooth)*nll_loss + smooth * smooth_loss
    if ignored is not None:
        loss = loss*ignored
    return loss.mean()

class ParaMultiOutputPoolerModel(pl.LightningModule):

    def __init__(self, bert_model, flag_lab2i, smooth=0, steps_train=None, weights=None, **args):
        super().__init__()
        self.save_hyperparameters()
        self.steps_train = steps_train
        self.weights = weights
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.flag_lab2i = flag_lab2i
        class_nums = {k: len(v) for k, v in self.flag_lab2i.items()}
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, n) for name, n in class_nums.items()})
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.smooth = smooth

    def forward(self,batch):
        enc=self.bert(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'],
                      token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        return {name: layer(enc.pooler_output) for name, layer in self.cls_layers.items()}

    def training_step(self,batch,batch_idx):
        out = self(batch)
        base_loss = smooth_loss(out['base'], batch['base'], self.smooth)
        flag_losses = [smooth_loss(out[k], batch[k], self.smooth, ignored=batch['is_4']) for k in ['direction', 'has_i', 'has_s']]
        self.accuracy(*encode_tensors(torch.stack(list(model_output_to_p(out).values())), torch.stack([batch[k] for k in ['base', 'direction', 'has_i', 'has_s']])))
        self.log("train_acc", self.accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return base_loss + sum(flag_losses)

    def validation_step(self,batch,batch_idx):
        out = self(batch)
        base_loss = smooth_loss(out['base'], batch['base'], self.smooth)
        flag_losses = [smooth_loss(out[k], batch[k], self.smooth, ignored=batch['is_4']) for k in ['direction', 'has_i', 'has_s']]
        self.log("val_loss", base_loss + sum(flag_losses), prog_bar=True)
        encoded = encode_tensors(torch.stack(list(model_output_to_p(out).values())), torch.stack([batch[k] for k in ['base', 'direction', 'has_i', 'has_s']]))
        self.val_accuracy(*encoded)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=1e-5)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.steps_train*0.1), num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

class ParaMultiOutputAvgModel(ParaMultiOutputPoolerModel):
 
    def __init__(self, **args):
        super().__init__(**args)
        self.save_hyperparameters()
        # self.drop_layer=torch.nn.Dropout(p=0.2)
        class_nums = {k: len(v) for k, v in args['flag_lab2i'].items()}
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size*5, n) for name, n in class_nums.items()})

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
        # dropped = self.drop_layer(catenated)

        return {name: layer(catenated) for name, layer in self.cls_layers.items()}
