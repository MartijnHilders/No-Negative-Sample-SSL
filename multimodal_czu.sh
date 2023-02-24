# CMC, czu
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc/cmc_inertial_skeleton_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/czu_mhad-mm_ssl_cmc_transformer_cooccurrence-2023-02-13_19_33_17_705269/last.ckpt \
    --res_json_path  /home/I6169337/semi_supervised_results/czu/mm_czu_cmc.json \

#CMC-CMKM
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc_cmkm/cmc_cmkm_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/czu_mhad-mm_ssl_cmc-cmkm_transformer_cooccurrence-2023-02-24_21_24_00_770191/last.ckpt \
    --res_json_path  /home/I6169337/semi_supervised_results/czu/mm_czu_cmc_cmkm.json \

# Barlow Twins preserved
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc/inertial_skeleton_barlow_preserved_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/czu_mhad-mm_ssl_barlow_transformer_cooccurrence-2023-02-24_16_03_45_322422/last.ckpt \
    --res_json_path  /home/I6169337/semi_supervised_results/czu/mm_czu_barlow.json \


# VICReg preserved.
python -m semi_supervised_evaluation \
    --experiment_config_path configs/cmc/vicreg_inertial_skeleton_preserved_czu.yaml \
    --dataset czu_mhad \
    --data_path /home/data/multimodal_har_datasets/czu_mhad  \
    --modalities inertial skeleton \
    --models transformer cooccurrence \
    --mode ssl \
    --fusion cmc \
    --pre_trained_paths  /home/I6169337/cmc/model_weights/czu_mhad-mm_ssl_vicreg_transformer_cooccurrence-2023-02-24_16_10_32_000216/last.ckpt \
    --res_json_path  /home/I6169337/semi_supervised_results/czu/mm_czu_vicreg.json \



