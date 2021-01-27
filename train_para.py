from sklearn.metrics import classification_report
from para_evaluate import evaluate
import torch
import argparse
import pytorch_lightning as pl
import para_data
import para_model
import para_averaging
import para_multi_output_model

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
    parser.add_argument('--epochs', default=3)

    args = parser.parse_args()
    load_checkpoint = args.load_checkpoint
    bert_path = args.bert_path
    label_strategy = args.label_strategy
    epochs = int(args.epochs)

    print(f"Using model: {args.model}, BERT from path: {bert_path}, label strategy: {label_strategy}, epochs {epochs}")

    batch_size = 8
    data=para_data.PARADataModule(".", batch_size, bert_model=bert_path, label_strategy=label_strategy)
    data.setup()
    size_train = len(data.train_data)
    steps_per_epoch = int(size_train/batch_size)
    steps_train = steps_per_epoch*epochs
    num_classes = len(data.train_data.label_encoder.classes_)
    # weights = torch.tensor(compute_class_weight('balanced', [l for l in range(4)], [t['label'] for t in data.train_data]))
    # print(f"Weights: {weights}")

    if args.model in ['multi-output-pooler', 'multi-output-averaging']:
        model_args = {'class_nums': {k: len(v) for k, v in data.train_data.flag_lab2i.items()}}
        
        inv = {n: {v: k for k, v in d.items()} for n, d in data.train_data.flag_lab2i.items()}

        def multi_output_to_pred(batch):
            str_batch = {k: [inv[k][e.item()] for e in v] for k, v in para_multi_output_model.model_output_to_p(batch).items()}
            label_batch = [''.join([b,
                                    d if d is not 'None' and b is '4' else '',
                                    'i' if i and b is '4' else '',
                                    's' if s and b is '4' else ''])
                           for b, d, i, s in zip(str_batch['base'], str_batch['direction'], str_batch['has_i'], str_batch['has_s'])]
            
            return data.train_data.label_encoder.transform(label_batch)

        model_output_to_p = multi_output_to_pred
        if args.model == 'multi-output-pooler':
            model_class = para_multi_output_model.ParaMultiOutputPoolerModel
        else:
            model_class = para_multi_output_model.ParaMultiOutputAvgModel
    else:
        model_args = {'num_classes': num_classes}
        model_output_to_p = lambda x: x.argmax(-1)
        if args.model == 'pooler':
            model_class = para_model.PARAModel
        elif args.model == 'averaging':
            model_class = para_averaging.ParaAvgModel
        else:
            print(f"Unknown model: {args.model}")
            raise SystemExit
    
    if load_checkpoint:
        model = model_class.load_from_checkpoint(bert_model=bert_path, steps_train=steps_train, checkpoint_path=load_checkpoint, **model_args)
    else:
        model = model_class(bert_model=bert_path, steps_train=steps_train, **model_args)

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
    evaluate(data, model, model_output_to_p)

    best_model = model_class.load_from_checkpoint(bert_model=bert_path, checkpoint_path=checkpoint_callback.best_model_path, **model_args)
    best_model.eval()
    best_model.cuda()
    evaluate(data, best_model, model_output_to_p)
