import multiprocessing
from typing import List

from pydantic import BaseSettings

import os 

__all__ = ["AudioSettings", "DatasetSettings", "SETTINGS"]


class ResourceSettings(BaseSettings):
    """Base settings for computational resources"""

    cpu_count: int = max(multiprocessing.cpu_count() // 2, 1)


class CacheSettings(BaseSettings):
    """Base settings for cache"""

    cache_size: int = 128144


class AudioSettings(BaseSettings):
    """Base settings for audio"""

    sample_rate: int = 16000
    use_mono: bool = True


class AudioTransformSettings(BaseSettings):
    """Base settings for audio transform"""

    num_fft: int = 512
    num_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 200
    use_meyda_spectrogram: bool = False


class InferenceEngineSettings(BaseSettings):
    """Base settings for inference engine"""
    # 下属为推理引擎的基本设置，实际参数如inference_sequence，inference_threshold使用
    # engine.set_threshold(float(current_threshold))
    # engine.set_sequence(current_sequence) 等方法动态设置

    inference_weights: List[float] = None
    inference_sequence: List[int] = [0,1]
                                    # 推理序列。这个序列表示系统会在这些帧中进行检测。
    
    inference_window_ms: float = 2000  # look at last of these seconds
                                    # 推理窗口的长度，以毫秒为单位。
                                    # 系统会在这个时间窗口内进行检测。例如，2000 毫秒表示
                                    # 系统会在最近的 2 秒内进行检测。
    
    smoothing_window_ms: float = 50  # prediction smoothed
    tolerance_window_ms: float = 500  # negative label between words
                                    # 容忍窗口的长度，以毫秒为单位。在这个时间窗口内，如果检测到负标签
                                    # （即非唤醒词），系统会容忍并继续检测。例如，500 毫秒表示
                                    # 系统会在最近的 500 毫秒内容忍负标签。

    inference_threshold: float = 0  # positive label probability must rise above this threshold
                                    # 推理阈值。正标签（即唤醒词）的概率必须超过这个阈值，
                                    # 系统才会认为检测到了唤醒词。例如，阈值为 0.5 表示
                                    # 系统只有在检测到唤醒词的概率超过 50% 时，才会认为检测到了唤醒词。


class TrainingSettings(BaseSettings):
    """Base settings for training"""

    seed: int = 0
    # TODO:: vocab should not belong to training
    vocab: List[str] = ["fire"]
    num_epochs: int = 10
    num_labels: int = 2
    learning_rate: float = 1e-3
    device: str = "cuda:0"
    batch_size: int = 16
    lr_decay: float = 0.955
    max_window_size_seconds: float = 0.75
    eval_window_size_seconds: float = 0.75
    eval_stride_size_seconds: float = 0.063
    weight_decay: float = 0
    convert_static: bool = False
    objective: str = "frame"  # frame or ctc
    # TODO: support phone token_type
    token_type: str = "word"
    phone_dictionary: str = None
    use_noise_dataset: bool = False
    noise_dataset_path: str = None


class DatasetSettings(BaseSettings):
    """Base settings for dataset"""

    dataset_path: str = None


class HowlSettings:
    """Lazy-loaded class containing all required settings"""

    _resource: ResourceSettings = None
    _audio: AudioSettings = None
    _audio_transform: AudioTransformSettings = None
    _inference_engine: InferenceEngineSettings = None
    _dataset: DatasetSettings = None
    _cache: CacheSettings = None
    _training: TrainingSettings = None

    @property
    def resource(self) -> ResourceSettings:
        """resource settings"""
        if self._resource is None:
            self._resource = ResourceSettings()
        return self._resource

    @property
    def audio(self) -> AudioSettings:
        """audio settings"""
        if self._audio is None:
            self._audio = AudioSettings()
        return self._audio

    @property
    def audio_transform(self) -> AudioTransformSettings:
        """audio transform settings"""
        if self._audio_transform is None:
            self._audio_transform = AudioTransformSettings()
        return self._audio_transform

    @property
    def inference_engine(self) -> InferenceEngineSettings:
        """inference engine settings"""
        if self._inference_engine is None:
            self._inference_engine = InferenceEngineSettings()
        return self._inference_engine

    @property
    def dataset(self) -> DatasetSettings:
        """dataset settings"""
        if self._dataset is None:
            self._dataset = DatasetSettings()
        return self._dataset

    @property
    def cache(self) -> CacheSettings:
        """cache settings"""
        if self._cache is None:
            self._cache = CacheSettings()
        return self._cache

    @property
    def training(self) -> TrainingSettings:
        """training settings"""
        if self._training is None:
            self._training = TrainingSettings()
        return self._training

    def reset(self):
        """Reset all the settings to defaults"""
        for attr_name, setting_cls in KEY_TO_SETTINGS_CLASS.items():
            setattr(self, attr_name, setting_cls())

    def __repr__(self):
        """Prints the contents of the settings in human-readable format"""
        rep = "Howl Settings:\n"
        for attr_name, settings in self.__dict__.items():
            rep += f"\t{attr_name}:"
            if settings is None:
                rep += " None\n"
            else:
                rep += "\n"
                for key, val in settings.__dict__.items():
                    rep += f"\t\t{key}: {val}\n"

        return rep


KEY_TO_SETTINGS_CLASS = {
    "_audio": AudioSettings,
    "_audio_transform": AudioTransformSettings,
    "_inference_engine": InferenceEngineSettings,
    "_dataset": DatasetSettings,
    "_cache": CacheSettings,
    "_training": TrainingSettings,
    "_resource": ResourceSettings,
}

SETTINGS = HowlSettings()
