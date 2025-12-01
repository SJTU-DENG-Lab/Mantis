# Available task_suite_name options: libero_spatial, libero_object, libero_goal, libero_10
# Available model_id options: Yysrc/LIBERO-Spatial, Yysrc/LIBERO-Object, Yysrc/LIBERO-Goal, Yysrc/LIBERO-Long or your own model path
# Available eval_mode options: action_chunking, temporal_ensemble, adaptive_temporal_ensemble

python experiments/libero/run_libero_eval.py \
    --task_suite_name libero_spatial \
    --model_family mantis \
    --model_id Yysrc/LIBERO-Spatial \
    --run_id_note None \
    --use_wandb False \
    --eval_mode temporal_ensemble \
    --dynamic_patches_threshold 0.12 \
    --target_patches_threshold 0.01 \
    --checkpoints_dir "" \
    --local_log_dir experiments/libero_eval_logs \
    --norm_file_path configs/norm_stats.json
