CUDA_VISIBLE_DEVICES=0 python3 src/reseval.py \
                      --algorithm $1 \
                      --eval_episodes 100 \
                      --seed $2 \
                      --eval_mode $3 \
                      --action_repeat 2 \
                      --domain_name $4 \
                      --task_name $5