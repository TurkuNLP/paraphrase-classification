import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch

def model_output_to_p(out):
    return {k: v.argmax(-1) for k, v in out.items()}

class ParaMultiOutputModel(pl.LightningModule):

    def __init__(self, bert_model, class_nums, steps_train=None, weights=None):
        super().__init__()
        self.steps_train = steps_train
        self.weights = weights
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, n) for name, n in class_nums.items()})
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self,batch):
        enc=self.bert(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'],
                      token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        return {name: layer(enc.pooler_output) for name, layer in self.cls_layers.items()}

    def training_step(self,batch,batch_idx):
        out = self(batch)
        base_loss = F.cross_entropy(out['base'], batch['base'], weight=None if not self.weights else self.weights[k].type_as(out))
        flag_losses = [F.cross_entropy(out[k], batch[k], weight=None if not self.weights else self.weights[k].type_as(out), reduction='none') for k in ['direction', 'has_i', 'has_s']]
        mean_losses = [torch.mean(t*batch['is_4']) for t in flag_losses]
        self.accuracy(torch.stack(list(model_output_to_p(out).values())), torch.stack([batch[k] for k in ['base', 'direction', 'has_i', 'has_s']]))
        self.log("train_acc", self.accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return base_loss + sum(mean_losses)

    def validation_step(self,batch,batch_idx):
        out = self(batch)
        base_loss = F.cross_entropy(out['base'], batch['base'], weight=None if not self.weights else self.weights[k].type_as(out))
        flag_losses = [F.cross_entropy(out[k], batch[k], weight=None if not self.weights else self.weights[k].type_as(out), reduction='none') for k in ['direction', 'has_i', 'has_s']]
        mean_losses = [torch.mean(t*batch['is_4']) for t in flag_losses]
        self.log("val_loss", base_loss + sum(mean_losses))
        self.val_accuracy(torch.stack(list(model_output_to_p(out).values())), torch.stack([batch[k] for k in ['base', 'direction', 'has_i', 'has_s']]))
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=1e-5)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.steps_train*0.1), num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]