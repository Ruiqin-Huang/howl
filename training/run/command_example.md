> 关于--load-weights参数的说明：在指定了--eval（仅进行[评估]时），不需要单独指定--load-weights参数，会默认自动加载模型权重
> 只有在[继续训练]的情况下在需要单独指定--load-weights参数，加载之前保存的模型权重
1. exp_hey_fire_fox_res8 20250227 仅训练
```bash
python -m training.run.eval_wake_word_detection --num_models 1 --seed 1919810 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'train'
```

2. exp_hey_fire_fox_res8 20250227 仅评估
```bash
python -m training.run.eval_wake_word_detection --num_models 1 --seed 1919810 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval'
```

3. exp_hey_fire_fox_res8 20250227 withParallelEval实验
```bash
python -m training.run.eval_wake_word_detection_withParallelEval --num_models 1 --seed 1919810 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval'
```

