from sklearn.metrics import classification_report
import torch
import argparse
import pytorch_lightning as pl
import para_model
import para_data

# Saves both the most accurate evaluated model as well as the last evaluated model.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints',
    filename='para-val_acc-max',
    save_top_k=1,
    mode='max',
    save_last=True
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    
    args = parser.parse_args()
    load_checkpoint = args.load_checkpoint

    data=para_data.PARADataModule(".",16)
    
    if load_checkpoint:
        model = para_model.PARAModel.load_from_checkpoint(checkpoint_path=load_checkpoint)
    else:
        model = para_model.PARAModel()

    trainer = pl.Trainer(
        resume_from_checkpoint=load_checkpoint,
        gpus=1,
        val_check_interval=0.25,
        num_sanity_val_steps=5,
        max_epochs=4,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data)

    model.eval()
    model.cuda()

    with torch.no_grad():
        preds = []
        for batch in data.val_dataloader():
            output = model(batch['input_ids'].cuda(), batch['token_type_ids'].cuda(), batch['attention_mask'].cuda()).argmax(-1)
            preds.append(output)

    preds = [e.item() for t in preds for e in t]

    eval_labels = [x['label'] for x in data.dev_data]

    print(classification_report(eval_labels, preds, digits=4))
