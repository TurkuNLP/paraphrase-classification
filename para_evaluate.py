from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def cos_sim(dataset):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    txt1 = [t['txt1'] for t in dataset.data_list]
    txt2 = [t['txt2'] for t in dataset.data_list]
    combined = vectorizer.fit(txt1 + txt2)
    tfidf_txt1 = torch.tensor(combined.transform(txt1).toarray())
    tfidf_txt2 = torch.tensor(combined.transform(txt2).toarray())
    return [torch.dot(t1, t2).item() for t1, t2 in zip(tfidf_txt1, tfidf_txt2)]

def evaluate(dataloader, dataset, model, model_output_to_p, save_directory=None):
    with torch.no_grad():
        preds = []
        for batch in dataloader:
            output = model_output_to_p(model({k: v.cuda() for k, v in batch.items()}))
            preds.append(output)

    preds = [e.item() for t in preds for e in t]
    target = [x['label'] for x in dataset]
    label_classes = dataset.label_encoder.classes_
    sim = cos_sim(dataset)
    
    str_preds = dataset.label_encoder.inverse_transform(preds)
    str_target = dataset.label_encoder.inverse_transform(target)
    i_preds = ['i' in p for p in str_preds]
    i_target = ['i' in p for p in str_target]
    s_preds = ['s' in p for p in str_preds]
    s_target = ['s' in p for p in str_target]
    coarse_preds = [p.replace('i', '').replace('s', '') for p in str_preds]
    coarse_target = [p.replace('i', '').replace('s', '') for p in str_target]
    coarse_classes = sorted([*{*coarse_target}])

    i_row = [e.item() for e in precision_recall_fscore_support(i_target, i_preds, labels=[True])]
    s_row = [e.item() for e in precision_recall_fscore_support(s_target, s_preds, labels=[True])]
    prec, recall, f1, support = [l.tolist() + [i] + [s] for l, i, s in zip(precision_recall_fscore_support(coarse_target, coarse_preds, labels=coarse_classes), i_row, s_row)]
    fprec = [f'{e:.4f}' for e in prec]
    frecall = [f'{e:.4f}' for e in recall]
    ff1 = [f'{e:.4f}' for e in f1]
    fsupport = [f'{e:d}' for e in support]
    rows = [list(l) for l in zip(*[coarse_classes + ['i'] + ['s'], fprec, frecall, ff1, fsupport])]
    rows = rows[:-2] + [[]] + rows[-2:]
    print('\t'.join(['', '', 'prec', 'recall', 'f1', 'support']))
    for row in rows:
        print('\t' + '\t'.join(row))
    print()
    print('\t'.join(['weighted avg', *[f'{e.item():.4f}' for e in precision_recall_fscore_support(str_target, str_preds, average='weighted')[:3]]]))
    print()

    if save_directory:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay(confusion_matrix(target, preds), display_labels=label_classes).plot(values_format='d')
    plt.savefig(path / "evaluation_confusion_matrix.pdf")
    print(classification_report(target, preds, target_names=label_classes, digits=4))

    pred_correctness = [p == l for p, l in zip(preds, target)]
    correct_sim = [s for s, p in zip(sim, pred_correctness) if p]
    incorrect_sim = [s for s, p in zip(sim, pred_correctness) if not p]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    # ax.set_title("Cosine similarity of sentence pairs in para_dev (Tf-Idf)")
    ax.boxplot([correct_sim, incorrect_sim], labels=["Correctly predicted", "Incorrectly predicted"])
    plt.show()
    if save_directory:
        plt.savefig(path / "cos_sim_boxplot.pdf")
    plt.close()

    data_dict = {'Cosine similarity': sim, 'Correctly predicted': pred_correctness}
    with sns.axes_style('whitegrid'), sns.color_palette('muted'):
        sns.displot(data=data_dict, x='Cosine similarity', hue='Correctly predicted', kind='kde', height=6, multiple='fill', clip=(0, 1))
        # plt.title("Conditional Density of Sentence Pairs in para_dev")
        plt.show()
        if save_directory:
            plt.savefig(path / "cos_sim_accuracy_density_plot.pdf")
        plt.close()
