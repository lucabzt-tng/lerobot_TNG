  uv run --active python -m rtc_server.inference_script \
      --model_path=/home/innovation-hacking/bozzetti/models/counterstrike/lerobot_pi05_test/050000/pretrained_model \
      --port=5555 --device=cuda --fps=60 \
      --execution_horizon=10 --max_guidance_weight=10.0 --prefix_attention_schedule=exp --compile