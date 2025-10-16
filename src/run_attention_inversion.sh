export HF_HOME="/playpen-nas-ssd/gongbang/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/playpen-nas-ssd/gongbang/.cache/huggingface/hub"

python edit_attention_inversion.py \
    --source_img_dir /playpen-nas-ssd/luchao/data/age/jackie_100/30_70 \
    --gender male \
    --ethnicity asian \
    --age_filter_young 27 33 \
    --age_filter_old 77 83 \
    --save_feature \
    --feature_path /playpen-nas-ssd/gongbang/try_code_features