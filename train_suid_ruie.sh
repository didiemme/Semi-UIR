DATASET=suid
python train.py --data_dir ./data/${DATASET} --save ./checkpoints/${DATASET}/ --log_dir ./logs/${DATASET} 
