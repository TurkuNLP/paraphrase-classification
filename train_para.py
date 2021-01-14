from sklearn.metrics import classification_report
from para_evaluate import evaluate
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

    epochs = 4
    batch_size = 16
    data=para_data.PARADataModule(".", batch_size)
    data.setup()
    size_train = len(data.train_data)
    steps_per_epoch = int(size_train/batch_size)
    steps_train = steps_per_epoch*epochs

    if load_checkpoint:
        model = para_model.PARAModel.load_from_checkpoint(checkpoint_path=load_checkpoint, steps_train=steps_train)
    else:
        model = para_model.PARAModel(steps_train=steps_train)

    trainer = pl.Trainer(
        resume_from_checkpoint=load_checkpoint,
        gpus=1,
        val_check_interval=0.25,
        num_sanity_val_steps=5,
        max_epochs=epochs,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data)

    model.eval()
    model.cuda()
    
    evaluate(data, model)
