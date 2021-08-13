# python -m src.common_path
# python -m src.super_prepare
# python -m src.lgb.lgb_prepare

# python -m src.lgb.lgb_train \
#   --learning_rate 0.01 \
#   --n_estimators 5000 \
#   --num_leaves 49 \
#   --subsample 0.65 \
#   --colsample_bytree 0.65 \
#   --random_state 2024

# python -m src.lgb.lgb_train \
#   --learning_rate 0.02 \
#   --n_estimators 6000 \
#   --num_leaves 49 \
#   --subsample 0.65 \
#   --colsample_bytree 0.65 \
#   --random_state 3000
  

#   #####################################################################
#   #
#   #                 nn training shell
#   #
#   #####################################################################
export CUDA_VISIBLE_DEVICES=0
start=$(date +%s)
for item_seed in 17 18 19 20 21
do
  python -m src.nn.train \
    --num_epochs 3 \
    --lr 0.01 \
    --bs 2048 \
    --ema_start_step 4000 \
    --logging_step 4000 \
    --l2_reg_embedding 0.1 \
    --l2 0.0001 \
    --dnn_inputs_dim 337 \
    --seed $item_seed
done
end=$(date +%s)
## 总训练时长（秒）
take=$(( end - start ))
take=$(( take/3600 ))
echo "总训练时长: ${take} h"

# export CUDA_VISIBLE_DEVICES=0
# start=$(date +%s)
# python -m src.nn.train \
#     --num_epochs 2 \
#     --lr 0.01 \
#     --bs 1024 \
#     --ema_start_step 4000 \
#     --logging_step 4000 \
#     --l2_reg_embedding 0.1 \
#     --l2 0.001 \
#     --dnn_inputs_dim 337 \
#     --seed 12
# end=$(date +%s)
# ## 总训练时长（秒）
# take=$(( end - start ))
# take=$(( take/3600 ))
# echo "总训练时长: ${take} h"

# python -m src.nn.train \
#     --num_epochs 1 \
#     --lr 0.01 \
#     --bs 25 \
#     --ema_start_step 10 \
#     --logging_step 10 \
#     --l2_reg_embedding 0.1 \
#     --l2 0.001 \
#     --dnn_inputs_dim 337 \
#     --seed 12 \
#     --debug_data
