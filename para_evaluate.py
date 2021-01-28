from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def evaluate(dataloader, dataset, model, model_output_to_p):
    with torch.no_grad():
        preds = []
        for batch in dataloader:
            output = model_output_to_p(model({k: v.cuda() for k, v in batch.items()}))
            preds.append(output)

    preds = [e.item() for t in preds for e in t]
    target = [x['label'] for x in dataset]
    label_classes = dataset.label_encoder.classes_
    sim = cos_sim(dataset)
    return preds, target, label_classes, sim

def print_results(preds, target, label_classes, sim, save_directory=None):
    if save_directory:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay(confusion_matrix(target, preds), label_classes).plot(values_format='d')
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
