mkdir -p imx500-implementation/logs

for model_dir in imx500-implementation/outputs/rpk/*; do
  model_name="$(basename "$model_dir")"
  python imx500-implementation/camera_imx500_live.py \
    --model "$model_dir/network.rpk" \
    --log-file "imx500-implementation/logs/${model_name}.jsonl" \
    --summary-file "imx500-implementation/logs/${model_name}_summary.json" \
    --frames 500
done