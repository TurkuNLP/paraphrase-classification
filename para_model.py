import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch

class PARAModel(pl.LightningModule):

    def __init__(self,bert_model="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__()
        self.bert=transformers.BertModel.from_pretrained(bert_model)
        self.cls_layer=torch.nn.Linear(768,4)
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self,input_ids,token_type_ids,attention_mask):
        enc=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids) #BxS_LENxSIZE; BxSIZE
        return self.cls_layer(enc.pooler_output)

    def training_step(self,batch,batch_idx):
        input_ids=batch["input_ids"]
        attention_mask=batch["attention_mask"]
        token_type_ids=batch["token_type_ids"]
        y_hat=self(input_ids,token_type_ids,attention_mask)
        loss=F.cross_entropy(y_hat,batch["label"])
        self.accuracy(y_hat,batch["label"])
        self.log("train_acc",self.accuracy,prog_bar=True,on_step=True,on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        input_ids=batch["input_ids"]
        attention_mask=batch["attention_mask"]
        token_type_ids=batch["token_type_ids"]
        y_hat=self(input_ids,token_type_ids,attention_mask)
        loss=F.cross_entropy(y_hat,batch["label"])
        self.log("val_loss",loss)
        self.val_accuracy(y_hat,batch["label"])
        self.log("val_acc",self.val_accuracy,prog_bar=True,on_epoch=True)
        
        
        

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=0.0000002)
        return optimizer
