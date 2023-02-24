# CMC, MMAct cross-scene
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc/cmc_inertial_skeleton_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --protocol cross_scene \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/.. \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xscene/mm_mmact-xscene_cmc.json \

#CMC-CMKM
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc_cmkm/cmc_cmkm_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/.. \
    --protocol cross_scene \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xscene/mm_mmact-xscene_cmc_cmkm.json \

# Barlow Twins preserved
python -m semi_supervised_evaluation \
    --experiment_config_path configs/barlow/inertial_skeleton_barlow_preserved_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/.. \
    --protocol cross_scene \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xscene/mm_mmact-xscene_barlow.json \


# VICReg preserved.
python -m semi_supervised_evaluation \
    --experiment_config_path configs/VICReg/inertial_skeleton_preserved_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths   /home/I6169337/cmc/model_weights/..    \
    --protocol cross_scene \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xscene/mm_mmact-xscene_vicreg.json \




