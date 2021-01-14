import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch

class PARAModel(pl.LightningModule):

    def __init__(self, steps_train=None, weights=None, bert_model="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__()
        self.steps_train = steps_train
        self.weights = weights
        self.bert=transformers.BertModel.from_pretrained(bert_model)
        self.cls_layer=torch.nn.Linear(768,4)
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self,batch):
        enc=self.bert(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'],
                      token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        return self.cls_layer(enc.pooler_output)

    def training_step(self,batch,batch_idx):
        y_hat=self(batch)
        loss=F.cross_entropy(y_hat, batch["label"], weight=None if not self.weights else self.weights.type_as(y_hat))
        self.accuracy(y_hat, batch["label"])
        self.log("train_acc", self.accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        y_hat=self(batch)
        loss=F.cross_entropy(y_hat, batch["label"], weight=None if not self.weights else self.weights.type_as(y_hat))
        self.log("val_loss", loss)
        self.val_accuracy(y_hat, batch["label"])
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=1e-5)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.steps_train*0.1), num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
