from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def cos_sim(data):
    vectorizer = TfidfVectorizer()
    txt1 = [t['txt1'] for t in data.dev_data.data_list]
    txt2 = [t['txt2'] for t in data.dev_data.data_list]
    combined = vectorizer.fit(txt1 + txt2)
    tfidf_txt1 = torch.tensor(combined.transform(txt1).toarray())
    tfidf_txt2 = torch.tensor(combined.transform(txt2).toarray())
    return [torch.dot(t1, t2).item() for t1, t2 in zip(tfidf_txt1, tfidf_txt2)]

def evaluate(data, model, model_output_to_p):
    with torch.no_grad():
        preds = []
        for batch in data.val_dataloader():
            output = model_output_to_p(model({k: v.cuda() for k, v in batch.items()}))
            preds.append(output)

    preds = [e.item() for t in preds for e in t]

    eval_labels = [x['label'] for x in data.dev_data]

    label_classes = data.dev_data.label_encoder.classes_
    ConfusionMatrixDisplay(confusion_matrix(eval_labels, preds), label_classes).plot(values_format='d')
    print(classification_report(eval_labels, preds, target_names=label_classes, digits=4))

    sim = cos_sim(data)
    pred_correctness = [p == l for p, l in zip(preds, eval_labels)]
    correct_sim = [s for s, p in zip(sim, pred_correctness) if p]
    incorrect_sim = [s for s, p in zip(sim, pred_correctness) if not p]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.set_title("Cosine similarity of sentence pairs in para_dev (Tf-Idf)")
    ax.boxplot([correct_sim, incorrect_sim], labels=["Correctly predicted", "Incorrectly predicted"])
    plt.show()
    plt.close()

    data_dict = {'Cosine Similarity': sim, 'Correctly Predicted': pred_correctness}
    with sns.axes_style('whitegrid'), sns.color_palette('muted'):
        sns.displot(data=data_dict, x='Cosine Similarity', hue='Correctly Predicted', kind='kde', height=6, multiple='fill', clip=(0, 1))
        plt.title("Conditional Density of Sentence Pairs in para_dev")
        plt.show()
        plt.close()

def cdplot_class(data):
    eval_labels_i = [x['label'] for x in data.dev_data]
    eval_labels = data.dev_data.label_encoder.inverse_transform(eval_labels_i)

    sim = cos_sim(data)

    data_dict = {'Cosine Similarity': sim, 'Label': eval_labels}
    with sns.axes_style('whitegrid'), sns.color_palette('muted'):
        sns.displot(data=data_dict, x='Cosine Similarity', hue='Label', kind='kde', height=6, multiple='fill', clip=(0, 1))
        plt.title("Conditional Density of Sentence Pairs in para_dev (by Label)")
        plt.show()
        plt.close()