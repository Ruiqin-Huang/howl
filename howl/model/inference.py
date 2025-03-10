import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from howl.context import InferenceContext
from howl.data.transform.operator import ZmuvTransform
from howl.data.transform.transform import StandardAudioTransform
from howl.model import RegisteredModel
from howl.settings import SETTINGS
from howl.utils import audio_utils
from howl.utils.logger import Logger

# TODO: add some imports
from typing import List

__all__ = ["FrameInferenceEngine", "InferenceEngine"]


class InferenceEngine:
    """Class of which handles inference using the trained model and context"""

    def __init__(
        self, model: RegisteredModel, zmuv_transform: ZmuvTransform, context: InferenceContext, time_provider=time.time
    ):
        """Initialize InferenceEngine

        Args:
            model (RegisteredModel): trained model
            zmuv_transform (ZmuvTransform): zmuv transformation to apply as a preprocessing step
            context (InferenceContext): context which describes how the inference should be done
            time_provider (time.time): module for current time
        """

        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.settings = SETTINGS.inference_engine
        self.context = context

        self.inference_weights = 1
        if self.settings.inference_weights:
            pad_size = context.num_labels - len(self.settings.inference_weights)
            self.inference_weights = np.pad(
                self.settings.inference_weights, (0, pad_size), "constant", constant_values=1
            )

        self.coloring = context.coloring
        self.negative_label = context.negative_label
        if self.coloring:
            self.negative_label = self.coloring.color_map[self.negative_label]

        self.sample_rate = SETTINGS.audio.sample_rate
        self.threshold = self.settings.inference_threshold
        self.inference_window_ms = self.settings.inference_window_ms
        self.smoothing_window_ms = self.settings.smoothing_window_ms
        self.tolerance_window_ms = self.settings.tolerance_window_ms
        self.sequence = self.settings.inference_sequence
        self.blank_idx = self.context.blank_label
        self.time_provider = time_provider

        self.curr_time = 0
        self.pred_history = []
        self.label_history = []
        self.reset()

    def to(self, device: torch.device):
        """Puts trained model and zmuv transformation model on GPU"""
        # pylint: disable=invalid-name
        self.model = self.model.to(device)
        self.zmuv = self.zmuv.to(device)
        return self

    def reset(self):
        """Reset stateful variables used for inference"""
        self.model.streaming_state = None
        self.curr_time = 0
        self.pred_history = []
        self.label_history = []

    def append_label(self, label: int, curr_time: float = None):
        """Append predicted label to label_history along with the current time

        Args:
            label (int): predicted label
            curr_time (float): current time
        """
        if curr_time is None:
            curr_time = self.time_provider() * 1000
        self.label_history.append((curr_time, label))

    def sequence_present(self, curr_time: float = None) -> bool:
        """Follow finite-state machine using detections in label_history
        to check if sequence of predicted labels make up the target sequence of words

        Args:
            curr_time (): current time

        Returns:
            return True if target sequence is present within the detection window
        """
        if not self.sequence:
            return False
        if len(self.sequence) == 0:
            return True

        if curr_time is None:
            curr_time = self.time_provider() * 1000

        self.label_history = list(
            itertools.dropwhile(lambda x: curr_time - x[0] > self.inference_window_ms, self.label_history)
        )  # drop entries that are old

        # finite state machine for detecting the sequence
        curr_label = None
        target_state = 0
        last_valid_timestamp = 0

        for history in self.label_history:
            curr_timestamp, label = history
            target_label = self.sequence[target_state]
            if label == target_label:
                # move to next state
                target_state += 1
                if target_state == len(self.sequence):
                    # goal state is reached
                    return True
                curr_label = self.sequence[target_state - 1]
                last_valid_timestamp = curr_timestamp
            elif label == curr_label:
                # label has not changed, only update last_valid_timestamp
                last_valid_timestamp = curr_timestamp
            elif last_valid_timestamp + self.tolerance_window_ms < curr_timestamp:
                # out of tolerance window, start from the first state
                curr_label = None
                target_state = 0
                last_valid_timestamp = 0
        return False

    def _get_prediction(self, curr_time: float) -> int:
        """Compute a prediction for the target window

        Args:
            curr_time (float): end timestamp for the window

        Returns:
            predicted label
        """
        # drop out-dated entries
        # 删除过期预测：移除所有超出平滑窗口(smoothing_window_ms)的历史预测
        # 保留最近时间窗口内的预测结果
        self.pred_history = list(
            itertools.dropwhile(lambda x: curr_time - x[0] > self.smoothing_window_ms, self.pred_history)
        )
        # 构建预测矩阵：将时间窗口内所有帧的预测概率垂直堆叠
        # 形成矩阵形状: [窗口内帧数 × 类别数]
        lattice = np.vstack([t for _, t in self.pred_history])
        # 获取最大概率：对每个类别，在时间窗口内取最大概率
        # 选择概率最高的类别作为预测标签
        lattice_max = np.max(lattice, 0)
        max_label = lattice_max.argmax()
        max_prob = lattice_max[max_label]
        if self.coloring:
            max_label = self.coloring.color_map.get(max_label, self.negative_label)
        # TODO: modify '<' to '<='
        if max_prob <= self.threshold:
            max_label = self.negative_label
        self.label_history.append((curr_time, max_label))
        return max_label

    def _append_probability_frame(self, prediction: np.ndarray, curr_time: float = None) -> int:
        """Append prediction (probability for each vocab) for a frame and get predicted label for the window

        Args:
            prediction (np.ndarray): prediction (probability for each vocab) for a frame
            curr_time (float): time of which the prediction is collected for

        Returns:
            predicted label
        """
        if curr_time is None:
            curr_time = self.time_provider() * 1000
        # 将当前帧的预测概率分布和时间戳添加到历史记录中（pred_history），供后续self._get_prediction等方法计算实际标签
        self.pred_history.append((curr_time, prediction))
        # 调用_get_prediction方法，计算当前时间窗口的预测标签
        return self._get_prediction(curr_time)

    # TODO: add set_threshold function
    def set_threshold(self, threshold: float):
        """Set inference threshold

        Args:
            threshold (float): new threshold value

        Returns:
            self: for method chaining
        """
        self.threshold = threshold
        return self
    
    # TODO: add set_sequence function
    def set_sequence(self, sequence: List[int]):
        """Set inference sequence for wake word detection

        Args:
            sequence (List[int]): New sequence to detect, e.g. [0,1] for two-word sequence

        Returns:
            self: for method chaining
        """
        self.sequence = sequence
        return self
    

    @torch.no_grad()
    def infer(self, audio_data: torch.Tensor) -> bool:
        """Checks if wake word is present in the given audio data;
        feed the whole audio data as a single batch

        Args:
            audio_data (torch.Tensor): audio data of which the wake word will be searched within

        Returns:
            return True if wake word presents in the last window
        """
        delta_ms = int(audio_data.size(-1) / self.sample_rate * 1000)
        self.std = self.std.to(audio_data.device)
        # TODO: compute lengths as in FrameInferenceEngine.ingest_frame
        transformed_frame = self.zmuv(self.std(audio_data.unsqueeze(0)))
        predictions = self.model(transformed_frame, lengths=None)  # [num_frames x 1 (batch size) x num_labels]
        predictions = F.softmax(predictions, -1).squeeze(1)  # [num_frames x num_labels]
        sequence_present = False
        delta_ms /= len(predictions)

        for prediction in predictions:
            prediction = prediction.cpu().numpy()
            prediction *= self.inference_weights
            prediction = prediction / prediction.sum()
            Logger.debug(([f"{probability:.3f}" for probability in prediction.tolist()], np.argmax(prediction)))
            self.curr_time += delta_ms
            if np.argmax(prediction) == self.blank_idx:
                continue
            self._append_probability_frame(prediction, curr_time=self.curr_time)
            if self.sequence_present(self.curr_time):
                sequence_present = True
                break

        return sequence_present


class FrameInferenceEngine(InferenceEngine):
    """InferenceEngine that evaluates the given audio data by generating predictions frame by frame"""

    def __init__(self, max_window_size_ms: int, eval_stride_size_ms: int, *args):
        """Initialize FrameInferenceEngine"""
        super().__init__(*args)
        self.max_window_size_ms, self.eval_stride_size_ms = max_window_size_ms, eval_stride_size_ms

    @torch.no_grad()
    def infer(self, audio_data: torch.Tensor) -> bool:
        """Checks if wake word is present in the given audio data;
        feed one frame at a time

        Args:
            audio_data (torch.Tensor): audio data of which the wake word will be searched within

        Returns:
            return True if wake word presents in the last window
        """
        sequence_present = False
        for window in audio_utils.stride(
            audio_data, self.max_window_size_ms, self.eval_stride_size_ms, self.sample_rate
        ):
            if window.size(-1) < 1000:
                break
            self.ingest_frame(window.squeeze(0), self.curr_time)
            self.curr_time += self.eval_stride_size_ms
            if self.sequence_present(self.curr_time):
                sequence_present = True
                break
        return sequence_present

    @torch.no_grad()
    def ingest_frame(self, frame: torch.Tensor, curr_time: float = None) -> int:
        """Feed the given frame of audio data and compute label

        Args:
            frame (torch.Tensor): frame of audio data
            curr_time (curr_time): current time

        Returns:
            predicted label
        """
        self.std = self.std.to(frame.device)
        lengths = torch.tensor([frame.size(-1)]).to(frame.device)
        transformed_lengths = self.std.compute_lengths(lengths)
        transformed_frame = self.zmuv(self.std(frame.unsqueeze(0)))
        prediction = self.model(transformed_frame, transformed_lengths).softmax(-1)[0].cpu().numpy()
        # 使用softmax函数将输出转换为概率

        # 将输出的概率乘以权重
        prediction *= self.inference_weights
        # 将输出的概率归一化，prediction是模型对当前帧的预测概率分布(numpy数组)
        prediction = prediction / prediction.sum()
        Logger.debug(([f"{probability:.3f}" for probability in prediction.tolist()], np.argmax(prediction)))
        label = self._append_probability_frame(prediction, curr_time=curr_time)
        return label
