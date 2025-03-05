# 生成模拟对齐的原理

> 参考代码：DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/howl/training/align/stub.py

执行代码为负集生成模拟对齐：
```bash
python -m training.run.attach_alignment \
  --input-raw-audio-dataset datasets/fire/negative \
  --token-type word \
  --alignment-type stub
```


1. 计算音频总长度（毫秒）
2. 将转录文本转为小写
3. 按字符级别在音频起点(0毫秒)到终点之间均匀分配时间戳


例如，对于样本{"path": "common_voice_en_19704228.wav", "transcription": "The network gained priority therein with regards to conflicts with his newspaper assignments."}

假设这个音频长度为5秒(5000毫秒)，这段转录文本包含98个字符(包括空格)
StubAligner会：
1. 计算每个字符的时间间隔：5000毫秒 ÷ 98个字符 ≈ 51毫秒/字符
2. 生成98个均匀分布的时间戳：[51, 102, 153, ..., 5000]
所以对于转录中的每个字符，包括"T"、"h"、"e"、" "(空格)、"n"等，都会分配到一个对应的时间戳。