import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data_modules.constants import DATASET_PROPERTIES
from models.mlp import UnimodalLinearEvaluator
from models.multimodal import MultiModalClassifier
from utils.training_utils import init_ssl_encoder, load_yaml_to_dict, init_transforms

def init_unimodal_finetuned(modality, model_cfg, dataset_cfg, finetuned_model_path):
    encoder = init_ssl_encoder(model_cfg)
    return UnimodalLinearEvaluator.load_from_checkpoint(finetuned_model_path, modality=modality,
                                                        encoder=encoder, in_size=encoder.out_size,
                                                        out_size=dataset_cfg['n_classes'], strict=False)

def init_multimodal_finetuned(modalities, model_cfgs, dataset_cfg, finetuned_model_path):
    encoders = {}
    for m in modalities:
        encoders[m] = init_ssl_encoder(model_cfgs[m])
    return MultiModalClassifier.load_from_checkpoint(finetuned_model_path, models_dict=encoders,
                                                     out_size=dataset_cfg['n_classes'], freeze_encoders=True)

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
    
    datamodule = DATASET_PROPERTIES[args.dataset].datamodule_class(modalities=args.modalities, train_transforms=train_transforms,
                                                                   test_transforms=test_transforms, split=dataset_cfg['protocols'][args.protocol])
    datamodule.setup()

    flatten = nn.Flatten()

    # Iterate over test data and gather ground truth labels.
    label_tensors = []
    for batch in datamodule.test_dataloader():
        label_tensors.append(batch['label'])
    all_labels = torch.cat(label_tensors)

    # Iterate over test data and gather predictions.
    if len(args.models) == 1:
        # SSL unimodal -> load the fine-tuned model.
        modality = args.modalities[0]
        model = init_unimodal_finetuned(modality, model_cfgs[modality], dataset_cfg, args.finetuned_model_path)
        model.freeze()

        prediction_tensors = []
        for batch in datamodule.test_dataloader():
            outs = model(batch[modality])
            preds = torch.argmax(outs, dim=1) + 1
            prediction_tensors.append(preds)
        
    else:
        # SSL multimodal (CMC or CMC-CVKM). Load the pre-trained multimodal classifier.
        model = init_multimodal_finetuned(args.modalities, model_cfgs, dataset_cfg, args.finetuned_model_path)
        model.freeze()

        prediction_tensors = []
        for batch in datamodule.test_dataloader():
            outs = model(batch)
            preds = torch.argmax(outs, dim=1) + 1
            prediction_tensors.append(preds)

    # Concatenate all predictions and save the results.
    all_predictions = torch.cat(prediction_tensors)
    labels_save_path = f"{args.save_path}_labels.pth"
    preds_save_path = f"{args.save_path}_preds.pth"
    torch.save(all_labels, labels_save_path)
    torch.save(all_predictions, preds_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--protocol', default="cross_subject")
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--finetuned_model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    args = parser.parse_args()

    main(args)