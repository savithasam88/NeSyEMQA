DATE="$(date +%m%d)"
SAVE_PATH="/users/sbsh670/archive/models" #"ckpt/Flash-VStream-Qwen-7b"
PYSCRIPT="/users/sbsh670/Flash-VStream/Flash-VStream-Qwen/eval_any_dataset.py"

ngpus=1
pixel_expr=4*224*224
max_pixels=$((${pixel_expr}))
max_frames=240


for dataset in mvbench 
do
    echo start eval ${dataset}
    python3 "${PYSCRIPT}" \
        --model-path ${SAVE_PATH} \
        --dataset ${dataset} \
        --output_dir ${SAVE_PATH} \
        --evaluation_name evaluation \
        --num_chunks ${ngpus} \
        --max_frames ${max_frames} \
        --max_pixels ${max_pixels} \
        >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}.log" 2>&1 
done
