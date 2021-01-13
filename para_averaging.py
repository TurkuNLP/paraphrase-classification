from para_evaluate import evaluate
import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch
import para_data

def transpose(l):
    return [list(t) for t in zip(*l)]

class ParaAvgModel(pl.LightningModule):

    def __init__(self, epochs, batch_size, size_train, weights=None, bert_model="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.size_train = size_train
        self.weights = weights
        self.bert=transformers.BertModel.from_pretrained(bert_model)
        # self.drop_layer=torch.nn.Dropout(p=0.2)
        self.cls_layer=torch.nn.Linear(768*5,4)
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, input_ids, token_type_ids, attention_mask, cls_mask, sep1_mask, sep2_mask, left_mask, right_mask):
        enc = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0] #BxS_LENxSIZE; BxSIZE
        cls = (enc*cls_mask.unsqueeze(-1)).sum(1) # enc.pooler_output
        sep1 = (enc*sep1_mask.unsqueeze(-1)).sum(1)
        sep2 = (enc*sep2_mask.unsqueeze(-1)).sum(1)
        left = (enc*left_mask.unsqueeze(-1)).sum(1) / left_mask.sum(-1).unsqueeze(-1)
        right = (enc*right_mask.unsqueeze(-1)).sum(1) / right_mask.sum(-1).unsqueeze(-1)
        catenated = torch.cat((cls, sep1, sep2, left, right), -1)
        # dropped = self.drop_layer(catenated)

        return self.cls_layer(catenated)

    def compute_masks(self, mask):
        one_idx = [i for i, b in enumerate(mask) if b]
        zeros = torch.zeros(len(mask), dtype=torch.long, device=self.device)
        cls_mask, sep1_mask, sep2_mask = [torch.scatter(zeros, 0, torch.tensor(i, device=self.device), torch.tensor(1, device=self.device)) for i in one_idx]
        left_idx = torch.tensor(range(one_idx[0]+1, one_idx[1]), device=self.device)
        left_mask = torch.scatter(zeros, 0, left_idx, torch.ones(len(left_idx), dtype=torch.long, device=self.device))
        right_idx = torch.tensor(range(one_idx[1]+1, one_idx[2]), device=self.device)
        right_mask = torch.scatter(zeros, 0, right_idx, torch.ones(len(right_idx), dtype=torch.long, device=self.device))
        return [cls_mask, sep1_mask, sep2_mask, left_mask, right_mask]

    def training_step(self,batch,batch_idx):
        y_hat = self(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"], batch["cls_mask"], batch["sep1_mask"], batch["sep2_mask"], batch["left_mask"], batch["right_mask"])
        loss = F.cross_entropy(y_hat, batch["label"], weight=None if not self.weights else self.weights.type_as(y_hat))
        self.accuracy(y_hat,batch["label"])
        self.log("train_acc",self.accuracy,prog_bar=True,on_step=True,on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        y_hat = self(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"], batch["cls_mask"], batch["sep1_mask"], batch["sep2_mask"], batch["left_mask"], batch["right_mask"])
        loss=F.cross_entropy(y_hat, batch["label"], weight=None if not self.weights else self.weights.type_as(y_hat))
        self.log("val_loss",loss)
        self.val_accuracy(y_hat,batch["label"])
        self.log("val_acc",self.val_accuracy,prog_bar=True,on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=1e-5)

        steps_per_epoch = int(self.size_train/self.batch_size)
        steps_train = steps_per_epoch*self.epochs
        steps_warmup = int(steps_train * 0.1)
        print(steps_warmup, steps_train)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_warmup, num_training_steps=steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints',
    filename='para-val_acc-max',
    save_top_k=1,
    mode='max',
    save_last=True
)

lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

if __name__=="__main__":
    epochs = 4
    batch_size = 16
    data = para_data.PARADataModule(".", batch_size)
    data.setup()
    size_train = len(data.train_data)
    # weights = torch.tensor(compute_class_weight('balanced', [l for l in range(4)], [t['label'] for t in data.train_data]))
    # print(f"Weights: {weights}")
    model = ParaAvgModel(epochs, batch_size, size_train)
    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=0.25,
        num_sanity_val_steps=5,
        max_epochs=epochs,
        progress_bar_refresh_rate=50,
        callbacks=[checkpoint_callback, lr_monitor_callback])
    trainer.fit(model,datamodule=data)

    model = ParaAvgModel.load_from_checkpoint(epochs=epochs, batch_size=batch_size, size_train=size_train, checkpoint_path=checkpoint_callback.last_model_path)
    model.eval()
    model.cuda()

    evaluate(data, model)

    best_model = ParaAvgModel.load_from_checkpoint(epochs=epochs, batch_size=batch_size, size_train=size_train, checkpoint_path=checkpoint_callback.best_model_path)
    best_model.eval()
    best_model.cuda()
    evaluate(data, best_model)
