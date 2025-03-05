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
4. exp_hey_fire_fox_res8 20250301 仅生成ROC曲线
```python
    ArgOption("--workspace", type=str, help="工作目录路径", required=True),
    ArgOption("--clean_report_name", type=str, help="干净数据集报告文件名(例如:hey_fire_fox_clean_20250228120442.xlsx)", required=True),
    ArgOption("--noisy_report_name", type=str, help="噪声数据集报告文件名(例如:hey_fire_fox_noisy_20250228120442.xlsx)", required=True),
    ArgOption("--total_dev_len", type=str, default="25739.7846", help="开发集音频总长度(秒)"),
    ArgOption("--total_test_len", type=str, default="24278.9884", help="测试集音频总长度(秒)"),
```
```bash
python -m training.run.generate_roc --workspace /home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/workspaces/exp_hey_fire_fox_res8 --clean_report_name hey_fire_fox_clean_20250303153618.xlsx --noisy_report_name hey_fire_fox_noisy_20250303153618.xlsx --total_dev_len "25739.7846" --total_test_len "24278.9884"
```

5. exp_hey_fire_fox_res8 20250301 正式训练，训练3个模型，仅训练 使用真实数据集
此次试验命名为exp-114514
或者exp-groundtruth-hey-fire-fox
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114514 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'train'
```

6. exp_hey_fire_fox_res8 20250302 评估3个模型，仅评估
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114514 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval'
```

7. exp_hey_fire_fox_res8 20250303 正式训练，训练3个模型，仅训练 使用stitched数据集
此次试验命名为exp-114515
或者exp-stitched-hey-fire-fox
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'train' --use-stitched-datasets
```

8. exp_hey_fire_fox_res8 20250304 评估3个模型，仅评估 使用stitched数据集训练的模型（442条拼接数据）
eid: exp-114515
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval' --use-stitched-datasets
```

9. exp_hey_fire_fox_res8 20250303 正式训练，训练3个模型，仅训练 使用4倍stitched数据集（442*4）
此次试验命名为exp-114515-4
或者exp-stitched-hey-fire-fox-4
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'train' --use-stitched-datasets
```

10. exp_hey_fire_fox_res8 20250303 评估3个模型，仅评估 使用4倍stitched数据集（442*4）
此次试验命名为exp-114515-4
或者exp-stitched-hey-fire-fox-4
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:0" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval' --use-stitched-datasets 
```

11. exp_hey_fire_fox_res8 20250304 评估3个模型，仅评估 使用stitched数据集训练的模型（442条拼接数据），将实验8中尚未评估结束的阈值[0.8, 0.85, 0.9, 0.95, 1.0]的结果进行评估
指定使用的阈值为[0.8, 0.85, 0.9, 0.95, 1.0]
指定--device为cuda:5，多进程在同一GPU上运并行评估
eid: exp-114515
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:5" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval' --use-stitched-datasets
```

12. exp_hey_fire_fox_res8 20250303 评估3个模型，仅评估 使用4倍stitched数据集（442*4）
此次试验命名为exp-114515-4
或者exp-stitched-hey-fire-fox-4
将实验8中尚未评估结束的阈值[1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45]的结果进行评估
指定使用的阈值为[1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45]
指定--device为cuda:5，多进程在同一GPU上运并行评估
```bash
python -m training.run.eval_wake_word_detection --num_models 3 --seed 114515 --exp_type "hey_fire_fox" --num_epochs "300" --vocab '["hey","fire","fox"]' --weight_decay "0.00001" --learning_rate "0.01" --lr_decay "0.98" --batch_size "16" --max_window_size_seconds "0.5" --eval_window_size_seconds "0.5" --eval_stride_size_seconds "0.063" --use_noise_dataset "True" --num_mels "40" --eval-freq "20" --device "cuda:5" --inference_sequence "[0,1,2]" --hop_size 0.05 --positive_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/positive" --negative_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl-train/datasets/hey_fire_fox/negative" --noise_dataset_path "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-noise/MS-SNSD" --exp_type 'eval' --use-stitched-datasets 
```