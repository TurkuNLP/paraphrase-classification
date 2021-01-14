from para_evaluate import evaluate
import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch
import para_data
import para_model

class ParaAvgModel(para_model.PARAModel):

    def __init__(self, **args):
        super().__init__(**args)
        # self.drop_layer=torch.nn.Dropout(p=0.2)
        self.cls_layer=torch.nn.Linear(768*5,4)

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

        return self.cls_layer(catenated)

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
    steps_per_epoch = int(size_train/batch_size)
    steps_train = steps_per_epoch*epochs
    # weights = torch.tensor(compute_class_weight('balanced', [l for l in range(4)], [t['label'] for t in data.train_data]))
    # print(f"Weights: {weights}")
    model = ParaAvgModel(steps_train=steps_train)
    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=0.25,
        num_sanity_val_steps=5,
        max_epochs=epochs,
        progress_bar_refresh_rate=50,
        callbacks=[checkpoint_callback, lr_monitor_callback])

    trainer.fit(model,datamodule=data)

    model.eval()
    model.cuda()
    evaluate(data, model)

    best_model = ParaAvgModel.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    best_model.eval()
    best_model.cuda()
    evaluate(data, best_model)
