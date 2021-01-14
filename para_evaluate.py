from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import torch

def evaluate(data, model):
    with torch.no_grad():
        preds = []
        for batch in data.val_dataloader():
            output = model({k: v.cuda() for k, v in batch.items()}).argmax(-1)
            preds.append(output)

    preds = [e.item() for t in preds for e in t]
    eval_labels = [x['label'] for x in data.dev_data]

    inv = {v: k for k, v in data.dev_data.lab2i.items()}
    labels = [inv[i] for i in range(4)]
    ConfusionMatrixDisplay(confusion_matrix(eval_labels, preds), labels).plot(values_format='d')
    print(classification_report(eval_labels, preds, target_names=labels, digits=4))
    
    vectorizer = TfidfVectorizer()
    txt1 = [t['txt1'] for t in data.dev_data.data_list]
    txt2 = [t['txt2'] for t in data.dev_data.data_list]
    combined = vectorizer.fit(txt1 + txt2)
    tfidf_txt1 = torch.tensor(combined.transform(txt1).toarray())
    tfidf_txt2 = torch.tensor(combined.transform(txt2).toarray())
    cos_sim = [torch.dot(t1, t2).item() for t1, t2 in zip(tfidf_txt1, tfidf_txt2)]
    pred_correctness = [p == l for p, l in zip(preds, eval_labels)]
    correct_sim = [s for s, p in zip(cos_sim, pred_correctness) if p]
    incorrect_sim = [s for s, p in zip(cos_sim, pred_correctness) if not p]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.set_title("Cosine similarity of sentence pairs in para_dev (Tf-Idf)")
    ax.boxplot([correct_sim, incorrect_sim], labels=["Correctly predicted", "Incorrectly predicted"])
    plt.show()
    plt.close()
