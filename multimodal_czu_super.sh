# random CZU
python -m semi_supervised_evaluation \
    --experiment_config_path configs/supervised/mm_inertial_skeleton_supervised_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode random \
    --res_json_path ../semi_supervised_results/multimodal/czu/mm_czu_random_fusion.json \

# supervised CZU
python -m semi_supervised_evaluation \
    --experiment_config_path configs/supervised/mm_inertial_skeleton_supervised_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode supervised \
    --res_json_path ../semi_supervised_results/multimodal/czu/mm_czu_supervised_fusion.json \






