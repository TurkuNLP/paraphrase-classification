from para_evaluate import evaluate
from para_classify_tsv import classify_tsv
from para_cluster_tsv import cluster_tsv
import torch
import argparse
import pytorch_lightning as pl
import para_data
import para_model
import para_averaging
import para_multi_output_model

lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--model', default='averaging')
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--label_strategy', default='coarse')
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--epochs', default=3)
    parser.add_argument('--evaluate', default=None)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--classify_tsv', nargs=2, default=None)
    parser.add_argument('--cluster_tsv', nargs=2, default=None)
    parser.add_argument('--json', nargs=3, default=['train.json', 'dev.json', 'test.json'])
    parser.add_argument('--model_out_dir', nargs=1, default=['checkpoints'])

    args = parser.parse_args()
    load_checkpoint = args.load_checkpoint
    bert_path = args.bert_path
    label_strategy = args.label_strategy
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    evaluate_set = args.evaluate
    no_train = args.no_train
    train_fname, dev_fname, test_fname = args.json
    model_out_dir = args.model_out_dir[0]

    print(f"Using model: {args.model}, BERT from path: {bert_path}, label strategy: {label_strategy}, batch size: {batch_size}, epochs: {epochs}", flush=True)
    if evaluate_set:
        print(f"Evaluating on {evaluate_set}", flush=True)
    else:
        print("No evaluation", flush=True)

    data=para_data.PARADataModule(batch_size, bert_model=bert_path, label_strategy=label_strategy, train_fname=train_fname, dev_fname=dev_fname, test_fname=test_fname)
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
            print(f"Unknown model: {args.model}", flush=True)
            raise SystemExit
    
    if load_checkpoint:
        model = model_class.load_from_checkpoint(bert_model=bert_path, steps_train=steps_train, checkpoint_path=load_checkpoint, **model_args)
    else:
        model = model_class(bert_model=bert_path, steps_train=steps_train, **model_args)

    # Saves both the most accurate evaluated model as well as the last evaluated model.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc',
        dirpath=model_out_dir,
        filename='para-val_acc-max',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    trainer = pl.Trainer(
        resume_from_checkpoint=load_checkpoint,
        gpus=1,
        val_check_interval=0.25,
        num_sanity_val_steps=5,
        max_epochs=epochs,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        callbacks=[checkpoint_callback]
    )

    if not no_train:
        trainer.fit(model, datamodule=data)

    if evaluate_set == 'dev':
        dataset = data.dev_data
        dataloader = data.val_dataloader()
    elif evaluate_set == 'test':
        dataset = data.test_data
        dataloader = data.test_dataloader()
    elif evaluate_set:
        print(f"Unknown evaluation set: {evaluate_set}", flush=True)
        raise SystemExit

    model.eval()
    model.cuda()
    if evaluate_set:
        evaluate(dataloader, dataset, model, model_output_to_p, save_directory='plots')
        
        if not no_train:
            best_model = model_class.load_from_checkpoint(bert_model=bert_path, checkpoint_path=checkpoint_callback.best_model_path, **model_args)
            best_model.eval()
            best_model.cuda()
            evaluate(dataloader, dataset, best_model, model_output_to_p, save_directory='plots')

    if args.classify_tsv:
        tsv_fname, tsv_out_fname = args.classify_tsv
        classify_tsv(model, bert_path, batch_size, tsv_fname, tsv_out_fname)

    if args.cluster_tsv:
        tsv_fname, tsv_out_fname = args.cluster_tsv
        cluster_tsv(model, bert_path, batch_size, label_strategy, tsv_fname, tsv_out_fname)
