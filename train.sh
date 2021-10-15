export MODEL_DIR=./pretrained_models/roformer
export DATA_DIR=./dataset
export OUTPUR_DIR=./outputs
export TASK_NAME=cmcc

#-----------training-----------------
python train.py \
  --model_type=roformer \
  --model_path=junnyu/roformer_chinese_base \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --gpu=0 \
  --monitor=eval_acc \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=1024 \
  --eval_max_seq_length=1024 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=20 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42