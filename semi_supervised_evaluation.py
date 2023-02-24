import argparse
import json
from pytorch_lightning import Trainer, seed_everything
import ssl_training
import ssl_training_mm
from supervised_training import train_test_supervised_model
from supervised_training_mm import train_test_supervised_mm_model

from utils.experiment_utils import (dict_to_json, generate_experiment_id,
                                    load_yaml_to_dict)
from utils.training_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # configs paths
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')

    # data and models
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--mode', default='ssl')
    parser.add_argument('--fusion', default='simple', choices=['simple', 'cmc', 'dtw'])
    parser.add_argument('--models', required=True, nargs='+')
    parser.add_argument('--pre_trained_paths', default=None, nargs='+')
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--model_save_path', default='./model_weights')
    parser.add_argument('--k_vals', nargs='+', default=[0.01, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument('--n_reps', default=10)
    parser.add_argument('--res_json_path')
    # other training configs
    parser.add_argument('--no_ckpt', action='store_true', default=True)
    
    return parser.parse_args()


def semi_supervised_training(args, cfg, dataset_cfg, k):
    experiment_id = generate_experiment_id()

    experiment_info = { 
        "dataset": args.dataset,
        "model": 'mm_' + '_'.join([cfg['modalities'][modality]['model'][args.models[i]]['class_name'] for i, modality in enumerate(args.modalities)]),
        "mode": args.mode,
        "k": k
    }

    if args.pre_trained_paths is not None:
        if len(args.pre_trained_paths) != len(args.modalities) != len(args.models):
                raise AttributeError('The numbers of pre-trained model paths, models and modalities should match.')
    if len(args.modalities) == 1:
        if args.pre_trained_paths is not None:
                pre_trained_path = args.pre_trained_paths[0]
        modality = args.modalities[0]
        model = args.models[0]
        args.model = model
    
    args.sweep = False
    args.framework = 'semi_sup'
    args.model_save_path = None
    args.num_workers = 8

    if args.mode == 'ssl':
        if args.pre_trained_paths is None:
            raise AttributeError('Model weights should be be provided.')
        
        if args.fusion in ['cmc', 'dtw']:
            # CMC or CMC-CVKM
            experiment_cfg = cfg['experiment']
            model_cfgs = {}
            transform_cfgs = {}
            for i, modality in enumerate(args.modalities):
                model_cfgs[modality] = cfg['modalities'][modality]['model'][args.models[i]]
                model_cfgs[modality]['kwargs'] = {**dataset_cfg[modality], **model_cfgs[modality]['kwargs']}
                transform_cfgs[modality] = cfg['modalities'][modality]['transforms']

            pre_trained_path = args.pre_trained_paths[0]
            pre_trained_model = init_ssl_mm_pretrained(args.modalities, model_cfgs, pre_trained_path, framework=args.fusion)
            encoders = getattr(pre_trained_model, 'encoders')

            loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality='_'.join(args.modalities), dataset=args.dataset, 
                experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, approach='semi_sup')

            args.framework = args.fusion
            args.no_ckpt = False

            metrics = ssl_training_mm.fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id, limited_k=k, random_seed=True)
        
        elif len(args.pre_trained_paths) == 1:
            pre_trained_path = args.pre_trained_paths[0]
            modality = args.modalities[0]
            model = args.models[0]

            model_cfg = cfg['modalities'][modality]['model'][model]
            model_cfg['kwargs'] = {**dataset_cfg[modality], **model_cfg['kwargs']}

            if pre_trained_path == "random":
                # Random encoder.
                encoder = init_ssl_encoder(model_cfg)
            else:
                # Unimodal SimCLR.
                pre_trained_model = init_ssl_pretrained(cfg['modalities'][modality]['model'][model], pre_trained_path)
                encoder = getattr(pre_trained_model, 'encoder')
                
            loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality=modality, dataset=args.dataset, 
                experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, approach='semi_sup')
            
            metrics = ssl_training.fine_tuning(args, modality, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id, limited_k=k)
        
        elif len(args.pre_trained_paths) == 2:
            # Simple fusion of encoders trained with SimCLR.
            args.ssl_pretrained = True
            metrics = train_test_supervised_mm_model(args, cfg, dataset_cfg, freeze_encoders=True, limited_k=k)
    else:
        if len(args.modalities) != len(args.models):
            raise AttributeError('The numbers of models and modalities should match.')
        elif len(args.modalities) == 1:
            # Supervised unimodal.
            metrics = train_test_supervised_model(args, cfg, dataset_cfg, freeze_encoder=True if args.mode == 'random' else False, 
                approach='semi_sup', experiment_info=experiment_info, limited_k=k)
        else:
            # Supervised multimodal (from scratch).
            args.ssl_pretrained = False
            args.no_ckpt = True
            metrics = train_test_supervised_mm_model(args, cfg, dataset_cfg, freeze_encoders=True if args.mode == 'random' else False, limited_k=k)
    return metrics


def main():
    args = parse_arguments()
    cfg = load_yaml_to_dict(args.experiment_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]

    result_dict = {}
    args.k_vals = [float(k) for k in args.k_vals]
    args.n_reps = int(args.n_reps)

    for k in args.k_vals:
        result_dict[k] = []
        for _ in range(args.n_reps):
            test_metrics = semi_supervised_training(args, cfg, dataset_cfg, k)
            result_dict[k].append(test_metrics)

    print(result_dict)
    dict_to_json(result_dict, args.res_json_path)


if __name__ == '__main__':
    main()