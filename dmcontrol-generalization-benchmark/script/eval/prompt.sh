CUDA_VISIBLE_DEVICES=0 python3 src/prompt_eval.py \
                      --algorithm $1 \
                      --prompt $2 \
                      --eval_episodes 100 \
                      --seed $3 \
                      --eval_mode $4 \
                      --action_repeat 2 \
                      --domain_name $5 \
                      --task_name $6