from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def cos_sim(data_list):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    txt1 = [t['txt1'] for t in data_list]
    txt2 = [t['txt2'] for t in data_list]
    combined = vectorizer.fit(txt1 + txt2)
    tfidf_txt1 = combined.transform(txt1).toarray()
    tfidf_txt2 = combined.transform(txt2).toarray()
    return [np.dot(t1, t2).item() for t1, t2 in zip(tfidf_txt1, tfidf_txt2)]

def parse_json(file_path):
    data_list = []
    with open(fname, 'r') as f:
        for entry in json.load(f):
            data_list.append({k: entry[k] for k in ['label', 'txt1', 'txt2']})
            for txt_pair in entry['rewrites']:
                data_list.append({'label': '4', 'txt1': txt_pair[0], 'txt2': txt_pair[1]})
    return data_list

def cdplot_class(sim, labels, path):
    label_classes = sorted([*{*labels}])
    data_dict = {'Cosine Similarity': sim, 'Label': labels}
    with sns.axes_style('whitegrid'), sns.color_palette('Paired', n_colors=len(label_classes)):
        sns.displot(data=data_dict, x='Cosine Similarity', hue='Label', hue_order=label_classes, kind='kde', height=6, multiple='fill', clip=(0, 1))
        plt.show()
        plt.savefig(path / 'cos_sim_label_density_plot.pdf')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="A JSON file containing the paraphrase entries")

    args = parser.parse_args()
    fname = args.file
    
    path = Path('plots')
    path.mkdir(parents=True, exist_ok=True)
    
    data_list = parse_json(fname)
    
    sim = cos_sim(data_list)
    labels = [x['label'] for x in data_list]
    print(f"Data set size: {len(data_list)} sentence pairs")
    print(f"Cosine similarity mean: {np.mean(sim):.4f}, stdev: {np.std(sim):.4f}")

    plt.hist(sim, histtype='stepfilled', bins=40)
    plt.show()
    plt.savefig(path / 'cos_sim_histogram.pdf')
    plt.close()

    cdplot_class(sim, labels, path)
