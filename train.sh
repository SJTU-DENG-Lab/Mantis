# Available config_file options: libero_spatial_image_action.yaml, libero_object_image_action.yaml, libero_goal_image_action.yaml, libero_long_image_action.yaml

torchrun --nproc-per-node=8 train.py \
    --run_name mantis_libero_spatial_image_action \
    --config_file libero_spatial_image_action.yaml \
    --base_dir ./ \
    --logging_dir log \
    > log/mantis_libero_spatial_image_action.log 2>&1