import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc

import torch
from sklearn.manifold import TSNE
import torch.nn.functional as F

from utils.training_utils import load_yaml_to_dict

def get_tsne_embeddings(embeddings, perplexity=50):
    embeddings = embeddings.flatten(end_dim=1)
    embeddings = F.normalize(embeddings, dim=1)
    embeddings_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(embeddings)
    return embeddings_2d

def get_labels(labels_path):
    labels = torch.load(labels_path, map_location=torch.device('cpu'))
    return labels

def draw_tsne_scatter(embeddings, labels, label_names, shapes, fig_name = './embeddings/res.pdf', legend_cols=2, title="T-SNE"):
    palette = sns.color_palette(cc.glasbey_dark, n_colors=len(labels.unique()))
    labels = [label_names[i] for i in labels]

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], style=shapes, hue=labels, alpha=0.9,
                   palette=palette).set_title(title)
    sns.set(font_scale=1.2)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., ncol=legend_cols)
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

def main(args):
    dataset_configs = load_yaml_to_dict('configs/dataset_configs.yaml')
    class_names = dataset_configs['datasets'][args.dataset]["class_names"]
    original_embeddings = torch.load(args.embeddings_path, map_location=torch.device('cpu'))

    # In the multimodal case, select only the desired embedding type.
    if isinstance(original_embeddings, dict):
        original_embeddings = original_embeddings[args.multimodal]

        # For projection head embeddings, concatenate them into a single tensor
        if args.multimodal == 'projections':
            projections = []
            for m in original_embeddings:
                projections.append(original_embeddings[m].unsqueeze(0))
            original_embeddings = torch.cat(projections)
        else:
            original_embeddings = original_embeddings.unsqueeze(0)

    embeddings = get_tsne_embeddings(original_embeddings, args.perplexity)

    n_modalities = original_embeddings.shape[0]

    labels_path = args.embeddings_path[:-15] + "_labels.pth"
    save_path = args.embeddings_path[:-15] + ".png"
    labels = get_labels(labels_path) - 1
    shapes = []
    for i in range(n_modalities):
        shapes.append(torch.full(labels.shape, i))
    shapes = torch.cat(shapes)
    labels = labels.repeat(n_modalities)
    print(shapes.shape)
    print(labels.shape)
    print(embeddings.shape)
    draw_tsne_scatter(embeddings, labels, class_names, shapes, fig_name=save_path, legend_cols=args.legend_columns, title=args.title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--multimodal', default='concatenated', choices=['projections', 'concatenated', 'inertial', 'skeleton'])
    parser.add_argument('--legend_columns', type=int, default=2)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--title', default="T-SNE")
    
    args = parser.parse_args()
    main(args)