export PYTHONPATH=$PYTHONPATH:../Downloads/carla_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg

CUDA_VISIBLE_DEVICES=0 python3 run_ppo.py train_policy_with_preferences carla-v0 --n_envs 1 --million_timesteps 1  \
  --root_dir logs \
  --experiment_name latent_sac \
  --gin_file params.gin \
  --gin_param load_carla_env.port=2000 \
  --load_prefs_dir=runs/merged