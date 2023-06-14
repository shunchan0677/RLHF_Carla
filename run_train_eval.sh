CUDA_VISIBLE_DEVICES=0 python run_interp.py \
  --root_dir logs \
  --experiment_name latent_sac \
  --gin_file params.gin \
  --gin_param load_carla_env.port=2000
