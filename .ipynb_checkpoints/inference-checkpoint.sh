# echo "LGB inference ..."

# python -m src.lgb.lgb_infer

# echo "NN inference ..."
# for item_seed in 12 13 14 15 16
# do
#   python -m src.nn.test \
#     --bs 512 \
#     --action '' \
#     --l2_reg_embedding 0.1 \
#     --l2 0.0001 \
#     --multi_modal_hidden_size 128 \
#     --seed $item_seed
# done

# python -m src.ensemble

# export CUDA_VISIBLE_DEVICES=0

# python -m src.nn.test \
#     --bs 1 \
#     --l2_reg_embedding 0.1 \
#     --l2 0.0001 \
#     --dnn_inputs_dim 340 \
#     --seed 12
    
    
# python -m src.nn.onnx_infer \
#     --bs 1 \
#     --l2_reg_embedding 0.1 \
#     --l2 0.0001 \
#     --dnn_inputs_dim 340 \
#     --seed 12

export CUDA_VISIBLE_DEVICES=0

# 清空submission目录
rm -r data/submission
mkdir -p data/submission

# 读取测试集路径
# test_path=./wbdc2021/data/wedata/wechat_algo_data2/test_a.csv
test_path=$1
echo "Test path: ${test_path}"
if [ "${test_path}" == "" ]; then
    echo "[Error] Please provide test_path!"
    exit 1
fi
test_size=$((`sed -n '$=' $test_path`-1))
echo "Test size: ${test_size}"
# 开始计时
start=$(date +%s)
# 调用模型预测代码
echo "NN inference ..."
for item_seed in 12 13 14 15 16 17 18 19 20 21
do
  ../envs/wbdc2021_onnx/bin/python -m src.nn.test \
    --bs 2048 \
    --l2_reg_embedding 0.1 \
    --l2 0.0001 \
    --dnn_inputs_dim 337 \
    --test_data_path ${test_path} \
    --seed $item_seed
done
echo "Start ensemble ..."
../envs/wbdc2021_onnx/bin/python -m src.ensemble
# 结束计时
end=$(date +%s)
# 计算耗时并输出
## 总预测时长（秒）
take=$(( end - start ))
echo "总预测时长: ${take} s"
## 单个目标行为2000条样本的平均预测时长（毫秒）
avg_take=$(echo "${take} ${test_size}"|awk '{print ($1*2000*1000/(7.0*$2))}')
echo "单个目标行为2000条样本的平均预测时长: ${avg_take} ms"