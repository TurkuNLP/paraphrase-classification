from sklearn.metrics import classification_report
from para_evaluate import evaluate
import torch
import argparse
import pytorch_lightning as pl
import para_data
import para_model
import para_averaging

# Saves both the most accurate evaluated model as well as the last evaluated model.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--model', default='averaging')
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--label_strategy', default='coarse')

    args = parser.parse_args()
    load_checkpoint = args.load_checkpoint
    if args.model == 'pooler':
        model_class = para_model.PARAModel
    else:
        model_class = para_averaging.ParaAvgModel
    bert_path = args.bert_path
    label_strategy = args.label_strategy

    print(f"Using model: {args.model}, with BERT from path: {bert_path}, and label strategy: {label_strategy}")

    epochs = 1
    batch_size = 16
    data=para_data.PARADataModule(".", batch_size, bert_model=bert_path, label_strategy=label_strategy)
    data.setup()
    size_train = len(data.train_data)
    steps_per_epoch = int(size_train/batch_size)
    steps_train = steps_per_epoch*epochs
    num_classes = len(data.train_data.label_encoder.classes_)
    # weights = torch.tensor(compute_class_weight('balanced', [l for l in range(4)], [t['label'] for t in data.train_data]))
    # print(f"Weights: {weights}")

    if load_checkpoint:
        model = model_class.load_from_checkpoint(bert_model=bert_path, num_classes=num_classes, steps_train=steps_train, checkpoint_path=load_checkpoint)
    else:
        model = model_class(bert_model=bert_path, num_classes=num_classes, steps_train=steps_train)

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

    best_model = model_class.load_from_checkpoint(bert_model=bert_path, num_classes=num_classes, checkpoint_path=checkpoint_callback.best_model_path)
    best_model.eval()
    best_model.cuda()
    evaluate(data, best_model)
