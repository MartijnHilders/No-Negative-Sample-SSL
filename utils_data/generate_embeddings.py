import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from data_modules.constants import DATASET_PROPERTIES
from utils.training_utils import init_ssl_mm_pretrained, init_ssl_pretrained, load_yaml_to_dict, init_transforms

def main(args):
    # Load configuration files.
    experiment_cfg = load_yaml_to_dict(args.config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]

    # Parse model configurations.
    model_cfgs = {}
    transform_cfgs = {}
    for i, m in enumerate(args.modalities):
        model_cfgs[m] = experiment_cfg["modalities"][m]["model"][args.models[i]]
        model_cfgs[m]['kwargs'] = {**dataset_cfg[m], **model_cfgs[m]['kwargs']}
        transform_cfgs[m] = experiment_cfg["modalities"][m]["transforms"]

    # Parse transform configurations.
    train_transforms = {}
    test_transforms = {}
    for m in args.modalities:
        cur_train_transforms, cur_test_transforms = init_transforms(m, transform_cfgs[m])
        train_transforms.update(cur_train_transforms)
        test_transforms.update(cur_test_transforms)

    # Initialise data module.
    data_split = {"train": {}, "test": {}} # do not filter data; test dataloader will contain entire dataset
    datamodule = DATASET_PROPERTIES[args.dataset].datamodule_class(modalities=args.modalities, train_transforms=train_transforms,
                                                                   test_transforms=test_transforms, split=data_split)
    datamodule.setup()

    flatten = nn.Flatten()

    # Iterate over dataset and gather labels.
    label_tensors = []
    for batch in datamodule.test_dataloader():
        label_tensors.append(batch['label'])
    all_labels = torch.cat(label_tensors)

    # Iterate over dataset and gather embeddings.
    if len(args.models) == 1:
        # SSL unimodal -> load the pre-trained encoder.
        pre_trained_model = init_ssl_pretrained(model_cfgs[args.modalities[0]], args.pretrained_path)
        encoder = getattr(pre_trained_model, 'encoder')
        encoder.freeze()

        # Use as embeddings the outputs of the encoder.
        modality = args.modalities[0]
        embedding_tensors = {modality : []}
        for batch in datamodule.test_dataloader():
            encoding = encoder(batch[modality])
            embedding_tensors[modality].append(flatten(encoding))
    else:
        # SSL multimodal (CMC or CMC-CVKM). Load the pre-trained encoder and the projection heads.
        pre_trained_model = init_ssl_mm_pretrained(args.modalities, model_cfgs, args.pretrained_path)
        encoders = pre_trained_model.encoders
        for m in encoders:
            encoders[m].freeze()
        projections = pre_trained_model.projections
        for m in projections:
            projections[m].freeze()
        
        final_embeddings = {}

        # Option 1: Use as embeddings the outputs of the projection heads.
        embedding_tensors = {m : [] for m in args.modalities}
        for batch in datamodule.test_dataloader():
            for m in args.modalities:
                encoding = encoders[m](batch[m])
                projection = projections[m](flatten(encoding))
                embedding_tensors[m].append(projection)
        final_embeddings['projections'] = {}
        for m in args.modalities:
            final_embeddings['projections'][m] = torch.cat(embedding_tensors[m])

        # Option 2: Normalize and concatenate the embeddings for each modality.
        m1, m2 = args.modalities[0], args.modalities[1]
        embedding_tensors = []
        for batch in datamodule.test_dataloader():
            # Modality 1
            encoding1 = encoders[m1](batch[m1])
            encoding1 = F.normalize(flatten(encoding1), dim=1)

            # Modality 2
            encoding2 = encoders[m2](batch[m2])
            encoding2 = F.normalize(flatten(encoding2), dim=1)

            # Concatenated
            concatenated = torch.cat([encoding1, encoding2], dim=1)
            embedding_tensors.append(concatenated)
        final_embeddings['concatenated'] = torch.cat(embedding_tensors)

        # Option 3: get the embedding for each individual modality.
        embedding_tensors = {m : [] for m in args.modalities}
        for batch in datamodule.test_dataloader():
            for m in args.modalities:
                encoding = encoders[m](batch[m])
                embedding_tensors[m].append(flatten(encoding))
        for m in args.modalities:
            final_embeddings[m] = torch.cat(embedding_tensors[m])

    # Save the results
    labels_save_path = f"{args.save_path}_labels.pth"
    embeddings_save_path = f"{args.save_path}_embeddings.pth"
    torch.save(all_labels, labels_save_path)
    torch.save(final_embeddings, embeddings_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    args = parser.parse_args()

    main(args)