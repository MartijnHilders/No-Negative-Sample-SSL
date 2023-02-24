# CMC, MMAct cross-subject
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc/cmc_inertial_skeleton_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/mmact-mm_ssl_cmc_transformer_cooccurrence-2023-02-13_13_29_55_080837/last.ckpt\
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xsubject/mm_mmact-xsubject_cmc.json \

#CMC-CMKM
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc_cmkm/cmc_cmkm_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/mmact-mm_ssl_cmc-cmkm_transformer_cooccurrence-2023-02-13_13_11_24_227689/last.ckpt\
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xsubject/mm_mmact-xsubject_cmc_cmkm.json \

# Barlow Twins preserved
python -m semi_supervised_evaluation \
    --experiment_config_path configs/barlow/inertial_skeleton_barlow_preserved_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/mmact-mm_ssl_barlow_transformer_cooccurrence-2023-02-24_15_36_55_071352/last.ckpt \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xsubject/mm_mmact-xsubject_barlow.json \


# VICReg preserved.
python -m semi_supervised_evaluation \
    --experiment_config_path configs/VICReg/inertial_skeleton_preserved_mmact.yaml \
    --dataset mmact \
    --data_path /home/data/multimodal_har_datasets/mmact_new \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths   /home/I6169337/cmc/model_weights/mmact-mm_ssl_vicreg_transformer_cooccurrence-2023-02-24_15_48_57_780036/last.ckpt    \
    --res_json_path  /home/I6169337/semi_supervised_results/mmact-xsubject/mm_mmact-xsubject_vicreg.json \




