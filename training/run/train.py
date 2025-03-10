from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm, trange

import numpy as np
import os
import time
from os import path
from typing import List
import logging

from howl.context import InferenceContext
from howl.data.common.tokenizer import WakeWordTokenizer
from howl.data.dataloader import StandardAudioDataLoaderBuilder
from howl.data.dataset.dataset import DatasetSplit, DatasetType, WakeWordDataset
from howl.data.dataset.dataset_loader import RecursiveNoiseDatasetLoader, WakeWordDatasetLoader
from howl.data.transform.batchifier import AudioSequenceBatchifier, WakeWordFrameBatchifier
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.data.transform.transform import (
    DatasetMixer,
    NoiseTransform,
    SpecAugmentTransform,
    StandardAudioTransform,
    TimeshiftTransform,
    TimestretchTransform,
)
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.model import ConfusionMatrix, ConvertedStaticModel, RegisteredModel
from howl.model.inference import FrameInferenceEngine, InferenceEngine
from howl.settings import SETTINGS
from howl.utils import hash_utils, random_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
from howl.utils.logger import setup_logger, Logger
from howl.workspace import Workspace

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import threading
import contextlib
import fcntl

def configure_logger(workspace_path: Path, action="train"):
    """配置训练日志系统"""
    # 创建日志目录
    log_dir = workspace_path / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 创建日志文件路径
    timestamp = time.strftime('%Y%m%d%H%M%S')
    if action == "eval":
        log_file = log_dir / f"model_{SETTINGS.training.seed}_eval_{timestamp}.log"
    else:
        log_file = log_dir / f"model_{SETTINGS.training.seed}_train_{timestamp}.log"

    # 初始化Logger
    Logger.LOGGER = setup_logger(
        name=Logger.NAME,
        level=logging.INFO,
        use_stdout=True,
        log_path=str(log_file)
    )
    
    return log_file
# Python 的 logging 模块会在每条日志后自动添加换行符
# 后续不需要在每次Logger.info .heading的时候手动添加换行符


# def acquire_file_lock(file_path):
#     """获取文件锁"""
#     lock_file = open(f"{file_path}.lock", 'w')
#     fcntl.flock(lock_file, fcntl.LOCK_EX)
#     return lock_file

# def release_file_lock(lock_file):
#     """释放文件锁"""
#     fcntl.flock(lock_file, fcntl.LOCK_UN)
#     lock_file.close()

# # 将函数移至全局作用域
# def evaluate_engine_threaded(
#     # 原有参数
#     dataset, prefix, positive_set, save, write_errors, mixer,
#     # 新增必要参数，这些原本是从main()作用域中获取的
#     use_frame, device, model, audio_transform, zmuv_transform, ctx, 
#     thresholds, args, SETTINGS, workspace, writer=None
# ):
#     """线程安全的评估引擎封装函数"""
#     Logger.info(f"=============== 数据集: {prefix} 开始评估[线程] ===============")
#     Logger.info(f"数据集大小: {len(dataset)}, 正样本集: {positive_set}, 写入错误: {write_errors}")
#     try:
#         Logger.info(f"[线程] 开始评估数据集: {prefix}")
#          # 线程安全地修改evaluate_engine内部的文件操作
#         audio_transform.eval()
        
#         if use_frame:
#             engine = FrameInferenceEngine(
#                 int(SETTINGS.training.max_window_size_seconds * 1000),
#                 int(SETTINGS.training.eval_stride_size_seconds * 1000),
#                 model,
#                 zmuv_transform,
#                 ctx,
#             )
#         else:
#             engine = InferenceEngine(model, zmuv_transform, ctx)
        
#         for current_threshold in thresholds:
#             engine.set_threshold(float(current_threshold))
#             model.eval()
#             conf_matrix = ConfusionMatrix()
        
#             # 使用线程安全的方式记录错误信息
#             if write_errors:
#                 lock_file = acquire_file_lock(workspace.path / "errors.tsv")
#                 try:
#                     with (workspace.path / "errors.tsv").open("a") as error_file:
#                         print(prefix, file=error_file)
#                 finally:
#                     release_file_lock(lock_file)
                        
#             pbar = tqdm(dataset, desc=f"{prefix} (阈值={current_threshold})")
#             for _, ex in enumerate(pbar):
#                 if mixer is not None:
#                     (ex,) = mixer([ex])
#                 audio_data = ex.audio_data.to(device)
#                 engine.reset()
#                 seq_present = engine.infer(audio_data)
                
#                 # 使用线程安全方式记录错误
#                 if seq_present != positive_set and write_errors:
#                     lock_file = acquire_file_lock(workspace.path / "errors.tsv")
#                     try:
#                         with (workspace.path / "errors.tsv").open("a", encoding='utf-8') as error_file:
#                             error_file.write(
#                                 f"{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\t阈值:{current_threshold}\n"
#                             )
#                     finally:
#                         release_file_lock(lock_file)
                            
#                 conf_matrix.increment(seq_present, positive_set)
#                 pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

#             Logger.info(f"{prefix} @ {engine.threshold}: {conf_matrix}")
#             Logger.info(f"在阈值 {engine.threshold}，推理序列: {engine.sequence}下的详细评估结果:")
#             Logger.info(f"真阳性(TP): {conf_matrix.tp}, 真阴性(TN): {conf_matrix.tn}")
#             Logger.info(f"假阳性(FP): {conf_matrix.fp}, 假阴性(FN): {conf_matrix.fn}")
            
#             if args.eval:
#                 threshold = engine.threshold
#                 lock_file = acquire_file_lock(workspace.path / (str(round(threshold, 2)) + "_results.csv"))
#                 try:
#                     with (workspace.path / (str(round(threshold, 2)) + "_results.csv")).open("a", encoding='utf-8') as result_file:
#                         result_file.write(
#                             f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n"
#                         )
#                 finally:
#                     release_file_lock(lock_file)

#         Logger.info(f"[线程] 完成评估数据集: {prefix}")
#         return True
        
#     except Exception as e:
#         Logger.error(f"[线程] 评估数据集 {prefix} 时出错: {str(e)}")
#         raise


# # nullcontext也移至全局作用域
# @contextlib.contextmanager
# def nullcontext():
#     yield

def main():
    """Train or evaluate howl model"""
    # TODO: train.py needs to be refactored
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=duplicate-code

    apb = ArgumentParserBuilder()
    apb.add_options(
        ArgOption("--model", type=str, choices=RegisteredModel.registered_names(), default="las",),
        ArgOption("--workspace", type=str, default=str(Path("workspaces") / "default")),
        ArgOption("--load-weights", action="store_true"), # 从之前保存的模型权重中加载模型参数，用于继续训练和模型评估
        # 关于--load-weights参数的说明：在指定了--eval（仅进行[评估]时），不需要单独指定--load-weights参数，会默认自动加载模型权重
        # 只有在[继续训练]的情况下在需要单独指定--load-weights参数，加载之前保存的模型权重
        ArgOption("--load-last", action="store_true"), # 若不指定，则加载最佳模型，否则加载最后一个epoch得到的模型
        ArgOption("--dataset-paths", "-i", type=str, nargs="+"), # 数据集路径
        ArgOption("--eval-freq", type=int, default=10),
        ArgOption("--eval", action="store_true"),
        ArgOption("--use-stitched-datasets", action="store_true"),
        ArgOption("--eval-hop-size", type=float, default=0.2),  # 添加 eval_hop_size 参数，默认值为 0.2
        ArgOption("--device", type=str, default="cuda:0"),  # 添加 device 参数，默认值为 "cuda:0"
    )
    args = apb.parser.parse_args()

    # region prepare training environment
    random_utils.set_random_seed(SETTINGS.training.seed)
    use_frame = SETTINGS.training.objective == "frame"
    workspace = Workspace(Path(args.workspace), delete_existing=not args.eval)
    # 当 args.eval=False（即不在评估模式运行）时，delete_existing=True，表示创建工作空间时会删除已存在的工作空间内容。
    # 当 args.eval=True（在评估模式运行）时，delete_existing=False，表示保留现有工作空间，不删除已有内容。
    writer = workspace.summary_writer
    
    # 设置日志系统
    if args.eval:
        log_file = configure_logger(workspace.path, action = "eval")
    else:
        log_file = configure_logger(workspace.path, action = "train")
    Logger.heading("Logging system initialized")
    Logger.info("=============== Logging System ===============")
    Logger.info(f"Log file created at: {log_file}")
    Logger.info(f"Workspace: {workspace.path}")
    
    device = torch.device(args.device)
    # set thresholds
    def round_and_convert_to_str(num):
        return str(round(num, 2))
    raw_thresholds = np.arange(0, 1.000001, args.eval_hop_size)
    thresholds = list(map(round_and_convert_to_str, raw_thresholds))
    # TODO: 设定为只取thresholds的最后n个值，即仅评估最后几个尚未评估完的阈值
    thresholds = thresholds[-5:] # 取最后5个阈值，即1.0 0.95 0.9 0.85 0.8
    # print the thresholds
    Logger.info(f"model will be evaluated with the following thresholds: {thresholds}")
    
    # 'cuda:0' represents the first available GPU device, 'cuda:1' represents the second available GPU device, and so on.
    Logger.info(f'Using device: {device}')    
    # verify GPU usage
    test_tensor = torch.randn(1)
    Logger.info(f"verifying device availability: {torch.cuda.is_available()}")
    Logger.info(f"Tensor before moving to device: {test_tensor.device}")
    test_tensor = test_tensor.to(device)
    Logger.info(f"Tensor after moving to device: {test_tensor.device}")
    
    # endregion prepare training environment

    # region load datasets
    Logger.heading("Loading datasets")
    ctx = InferenceContext(
        vocab=SETTINGS.training.vocab, token_type=SETTINGS.training.token_type, use_blank=not use_frame,
    )
    loader = WakeWordDatasetLoader()
    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=ctx.labeler,)

    ww_train_ds, ww_dev_ds, ww_test_ds = (
        WakeWordDataset(
            metadata_list=[], set_type=DatasetType.TRAINING, dataset_split=DatasetSplit.TRAINING, **ds_kwargs
        ),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.DEV, dataset_split=DatasetSplit.DEV, **ds_kwargs),
        WakeWordDataset(metadata_list=[], set_type=DatasetType.TEST, dataset_split=DatasetSplit.TEST, **ds_kwargs),
    )
    
    # 先加载常规数据集（aligned-开头）
    for ds_path in args.dataset_paths:
        ds_path = Path(ds_path)
        train_ds, dev_ds, test_ds = loader.load_splits(ds_path, **ds_kwargs)
        ww_train_ds.extend(train_ds)
        ww_dev_ds.extend(dev_ds)
        ww_test_ds.extend(test_ds)

    # 打印并保存统计信息
    train_stats = ww_train_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)
    dev_stats = ww_dev_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)
    test_stats = ww_test_ds.print_stats(word_searcher=ctx.searcher, compute_length=True)

    # 可选：将这些统计信息保存到全局变量或配置中，以便其他函数使用
    total_counts = {
        "Train positive": str(train_stats.num_examples),
        "Train length": str(train_stats.audio_length_seconds),
        "Dev positive": str(dev_stats.num_examples),
        "Dev length": str(dev_stats.audio_length_seconds),
        "Test positive": str(test_stats.num_examples),
        "Test length": str(test_stats.audio_length_seconds),
    }
    
    # TODO: 将得到的total_counts统计信息，特别是关于时间计算的信息写入../workspace/dataset/dataset_stats.json文件中
    # 以便eval_wake_word_detection.py中生成报告的时候可以直接读取这些统计信息

    if args.use_stitched_datasets:
        Logger.heading("Loading stitched datasets")
        ds_kwargs.pop("frame_labeler")
        ds_kwargs["labeler"] = ctx.labeler
        for ds_path in args.dataset_paths:
            ds_path = Path(ds_path)
            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.STITCHED, ds_path)
            try:
                # 当指定--use-stitched-datasets参数时，会加载stitched数据集
                # 并且，加载的stitched数据集不会覆盖已加载的常规数据集，而是叠加到同一个集合中。
                # TODO: 所以，如果希望只加载stitched数据集，需要把aligned-开头的文件内容暂时清空
                train_ds, dev_ds, test_ds = dataset_loader.load_splits(**ds_kwargs)
                ww_train_ds.extend(train_ds)
                ww_dev_ds.extend(dev_ds)
                ww_test_ds.extend(test_ds)
            except FileNotFoundError as file_not_found_error:
                Logger.error(f"Stitched dataset is missing for {ds_path}: {file_not_found_error}")

        header = "w/ stitched"
        train_stats_stitched = ww_train_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)
        dev_stats_stitched = ww_dev_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)
        test_stats_stitched = ww_test_ds.print_stats(header=header, word_searcher=ctx.searcher, compute_length=True)

        # 更新统计信息
        total_counts.update({
            "Train positive": str(train_stats_stitched.num_examples),
            "Train length": str(train_stats_stitched.audio_length_seconds),
            "Dev positive": str(dev_stats_stitched.num_examples),
            "Dev length": str(dev_stats_stitched.audio_length_seconds),
            "Test positive": str(test_stats_stitched.num_examples),
            "Test length": str(test_stats_stitched.audio_length_seconds),
        })

    ww_dev_pos_ds = ww_dev_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    ww_dev_pos_ds.print_stats(header="dev_pos", word_searcher=ctx.searcher, compute_length=True)
    ww_dev_neg_ds = ww_dev_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    ww_dev_pos_ds.print_stats(header="dev_neg", word_searcher=ctx.searcher, compute_length=True)
    ww_test_pos_ds = ww_test_ds.filter(lambda x: ctx.searcher.search(x.transcription), clone=True)
    ww_test_pos_ds.print_stats(header="test_neg", word_searcher=ctx.searcher, compute_length=True)
    ww_test_neg_ds = ww_test_ds.filter(lambda x: not ctx.searcher.search(x.transcription), clone=True)
    ww_test_neg_ds.print_stats(header="test_neg", word_searcher=ctx.searcher, compute_length=True)

    # endregion load datasets

    # region create dataloaders with audio preprocessor
    audio_transform = StandardAudioTransform().to(device).eval()
    zmuv_transform = ZmuvTransform().to(device)
    if use_frame:
        batchifier = WakeWordFrameBatchifier(
            ctx.negative_label, window_size_ms=int(SETTINGS.training.max_window_size_seconds * 1000),
        )
    else:
        tokenizer = WakeWordTokenizer(ctx.vocab, ignore_oov=False)
        batchifier = AudioSequenceBatchifier(ctx.negative_label, tokenizer)

    audio_augmentations = (
        TimestretchTransform().train(),
        TimeshiftTransform().train(),
        NoiseTransform().train(),
        batchifier,
    )

    if SETTINGS.training.use_noise_dataset:
        Logger.info("[DEBUG SETTINGS]Using noise dataset to augment...")
        noise_ds = RecursiveNoiseDatasetLoader().load(
            Path(SETTINGS.training.noise_dataset_path),
            sample_rate=SETTINGS.audio.sample_rate,
            mono=SETTINGS.audio.use_mono,
        )
        Logger.info(f"Loaded {len(noise_ds.metadata_list)} noise files.")
        noise_ds_train, noise_ds_dev = noise_ds.split(hash_utils.Sha256Splitter(80))
        noise_ds_dev, noise_ds_test = noise_ds_dev.split(hash_utils.Sha256Splitter(50))
        audio_augmentations = (DatasetMixer(noise_ds_train).train(),) + audio_augmentations
        dev_mixer = DatasetMixer(noise_ds_dev, seed=0, do_replace=False)
        test_mixer = DatasetMixer(noise_ds_test, seed=0, do_replace=False)
    audio_augmentations = compose(*audio_augmentations)

    prep_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=batchify).build(1)
    # TODO: decrease num_workers to save server resources
    # 默认num_workers=mp.cpu_count()，可以适当减少以节约服务器资源消耗
    prep_dl.shuffle = True
    train_dl = StandardAudioDataLoaderBuilder(ww_train_ds, collate_fn=audio_augmentations).build(
        SETTINGS.training.batch_size
    )
    # 默认num_workers=mp.cpu_count()，可以适当减少以节约服务器资源消耗
    # endregion initialize audio pre-processors

    # region prepare model for zmuv normalization
    Logger.heading("ZMUV normalization")
    if (workspace.path / "zmuv.pt.bin").exists():
        zmuv_transform.load_state_dict(torch.load(str(workspace.path / "zmuv.pt.bin")))
    else:
        for idx, batch in enumerate(tqdm(prep_dl, desc="Constructing ZMUV")):
            # 处理完整的数据集来计算ZMUV统计信息，从而获得更准确的标准化参数。
            batch.to(device)
            zmuv_transform.update(audio_transform(batch.audio_data))
        Logger.info(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str(workspace.path / "zmuv.pt.bin"))
    # endregion prepare model for zmuv normalization

    # region prepare model training
    Logger.heading("Model preparation")
    model = RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).streaming()
    if SETTINGS.training.convert_static:
        model = ConvertedStaticModel(model, 40, 10)

    if use_frame:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CTCLoss(ctx.blank_label)

    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, SETTINGS.training.learning_rate, weight_decay=SETTINGS.training.weight_decay,)
    Logger.info(f"{sum(p.numel() for p in params)} parameters")

    if args.load_weights:
        workspace.load_model(model, best=not args.load_last)
    # endregion prepare model training
    # --load-weights目的是从之前保存的模型权重中加载模型参数。这在以下几种情况下非常有用：
    # 继续训练： 如果你之前已经训练了一部分模型，并且希望从上次中断的地方继续训练，可以使用 load_weights 加载之前保存的模型权重。这可以避免从头开始训练，节省时间和计算资源。
    # 评估模型： 在评估模型时，你可能希望加载之前训练好的模型权重，以便在验证集或测试集上进行评估。这样可以确保评估的是训练好的模型，而不是随机初始化的模型。

    # evaluate_engine，仅用于训练过程中在dev集上评估模型，并执行workspace.increment_model更新最佳模型
    def evaluate_engine(
        dataset: WakeWordDataset,
        prefix: str,
        save: bool = False,
        # save 参数用于控制是否在验证过程中保存模型和记录指标。具体来说，当 save=True 时
        # 会在验证过程中保存模型: workspace.increment_model(model, conf_matrix.tp)
        # 并记录指标: writer.add_scalar
        # 否则只会记录指标，不会保存模型。
        # 如果当前模型的真阳性(TP)指标比之前保存的模型更好，就会更新保存的"最佳模型"。用于在训练过程中保存最佳模型model-best.pt.bin
        positive_set: bool = False,
        # positive_set 参数用于指示当前评估的数据集是否为正样本集（包含唤醒词的样本集）。
        # 当 positive_set=True 时，表示当前评估的样本应该包含唤醒词
        # 当 positive_set=False 时，表示当前评估的样本不应该包含唤醒词
        write_errors: bool = True,
        mixer: DatasetMixer = None,
    ):
        """Evaluate the current model on the given dataset"""
        Logger.info(f"=============== 数据集: {prefix} 开始评估 ===============")
        Logger.info(f"数据集大小: {len(dataset)}, 正样本集: {positive_set}, 写入错误: {write_errors}")
        audio_transform.eval()
        
        # 批量数据加载器
        # eval_batch_size = 8  # 根据GPU显存大小调整
        # eval_num_workers = 4  # 工作线程数
        # eval_dataloader = torch.utils.data.DataLoader(
        #     dataset, 
        #     batch_size=eval_batch_size,
        #     shuffle=False, 
        #     num_workers=eval_num_workers,
        #     pin_memory=True  # 加速CPU到GPU的数据传输
        # )
        

        if use_frame:
            engine = FrameInferenceEngine(
                int(SETTINGS.training.max_window_size_seconds * 1000),
                int(SETTINGS.training.eval_stride_size_seconds * 1000),
                model,
                zmuv_transform,
                # zmuv_transform是零均值单位方差(Zero Mean Unit Variance)变换
                # 它用于标准化音频特征，确保输入模型的数据分布稳定
                ctx,
                # ctx是推理上下文(InferenceContext)
                # 包含词汇表、标记类型和标签映射等基础配置
                # zmuv_transform和ctx都是一个与阈值无关的预处理步骤，更换阈值评估时不需要重新加载
            )
        else:
            engine = InferenceEngine(model, zmuv_transform, ctx)
        Logger.info(f"使用推理引擎: {engine.__class__.__name__}")
        # Logger.info(f"批处理大小: {eval_batch_size}")
        # Logger.info(f"工作线程数: {eval_num_workers}")
        Logger.info(f"最大窗口大小: {SETTINGS.training.max_window_size_seconds}秒")
        Logger.info(f"评估窗口大小: {SETTINGS.training.eval_window_size_seconds}秒")
        Logger.info(f"评估步长: {SETTINGS.training.eval_stride_size_seconds}秒")
        for current_threshold in thresholds:
            # os.environ["INFERENCE_THRESHOLD"] = current_threshold # deprecated
            # pydantic.BaseSettings 类通常只在创建对象时读取环境变量
            # 设置环境变量后，需要手动刷新 SETTINGS 或直接修改 engine.threshold
            # 在 InferenceEngine 初始化时，设置了 self.threshold = self.settings.inference_threshold
            # 后续没有重新获取或监听环境变量的变化
            engine.set_threshold(float(current_threshold))
            
            # 下面直至
            # with (workspace.path / (str(round(threshold, 2)) + "_results.csv")).open("a", encoding='utf-8') as result_file:
            #         result_file.write(
            #             f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n"
            #         )
            # 之间的代码块，向后缩进一个层级
            ###########################################
            model.eval()
            conf_matrix = ConfusionMatrix()
            # pbar-1
            pbar = tqdm(dataset, desc=prefix)
            # 单样本处理：直接从数据集一次获取一个样本
            # 串行执行：每次只处理一个音频样本
            # 进度条信息：只显示数据集名称（如"Dev positive"）
            # pbar = tqdm(eval_dataloader, desc=f"{prefix} (阈值={current_threshold})")
            # 批量处理：每次处理32个（或自定义数量）样本
            # 并行计算：充分利用GPU并行计算能力
            # 详细进度信息：同时显示数据集名称和当前阈值
            if write_errors:
                with (workspace.path / "errors.tsv").open("a") as error_file:
                    print(prefix, file=error_file)
            for _, ex in enumerate(pbar):
                # 一条一条音频数据进行推理
                if mixer is not None:
                    (ex,) = mixer([ex])
                audio_data = ex.audio_data.to(device)
                engine.reset()
                seq_present = engine.infer(audio_data)
                if seq_present != positive_set and write_errors:
                    # 当模型预测结果与真实标签不完全相符时（seq_present != positive_set），会记录这个错误样本的信息。
                    with (workspace.path / "errors.tsv").open("a", encoding='utf-8') as error_file:
                        error_file.write(
                            f"{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\t阈值:{current_threshold}\n"
                        )
                conf_matrix.increment(seq_present, positive_set)
                # seq_present为模型预测结果，positive_set为真实标签, 通过conf_matrix.increment更新混淆矩阵
                pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

            Logger.info(f"{conf_matrix}")
            Logger.info(f"在阈值 {engine.threshold}，推理序列: {engine.sequence}下的详细评估结果:")
            Logger.info(f"真阳性(TP): {conf_matrix.tp}, 真阴性(TN): {conf_matrix.tn}")
            Logger.info(f"假阳性(FP): {conf_matrix.fp}, 假阴性(FN): {conf_matrix.fn}")
            if save and not args.eval and positive_set:
                # 当 save=True 时，表示在验证过程中保存模型和记录指标。
                # 只有在 positive_set=True 时（即评估的是正样本集）才会考虑保存模型。这是因为唤醒词系统通常更关注模型在正确识别唤醒词方面的表现，而不是在负样本上的表现。
                writer.add_scalar(f"{prefix}/Metric/tp_rate", conf_matrix.tp / len(dataset), epoch_idx)
                workspace.increment_model(model, conf_matrix.tp)
            if args.eval:
                threshold = engine.threshold
                with (workspace.path / (str(round(threshold, 2)) + "_results.csv")).open("a", encoding='utf-8') as result_file:
                    result_file.write(
                        f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n"
                    )
        Logger.info(f"=============== 数据集: {prefix} 评估完成 ===============")


    # 并行评估
    def evaluate_engine_threaded(
        dataset: WakeWordDataset,
        prefix: str,
        positive_set: bool = False,
        # positive_set 参数用于指示当前评估的数据集是否为正样本集（包含唤醒词的样本集）。
        # 当 positive_set=True 时，表示当前评估的样本应该包含唤醒词
        # 当 positive_set=False 时，表示当前评估的样本不应该包含唤醒词
        save: bool = False,
        # save 参数用于控制是否在验证过程中保存模型和记录指标。具体来说，当 save=True 时
        # 会在验证过程中保存模型: workspace.increment_model(model, conf_matrix.tp)
        # 并记录指标: writer.add_scalar
        # 否则只会记录指标，不会保存模型。
        # 如果当前模型的真阳性(TP)指标比之前保存的模型更好，就会更新保存的"最佳模型"。用于在训练过程中保存最佳模型model-best.pt.bin
        write_errors: bool = True,
        mixer: DatasetMixer = None,
        file_lock: threading.Lock = None
    ):
        """Evaluate the current model on the given dataset"""
        Logger.info(f"=============== 数据集: {prefix} 开始评估[线程] ===============")
        Logger.info(f"数据集大小: {len(dataset)}, 正样本集: {positive_set}, 写入错误: {write_errors}")
        try:
            Logger.info(f"[线程] 开始评估数据集: {prefix}")
             # 线程安全地修改evaluate_engine内部的文件操作
            audio_transform.eval()
            
            if use_frame:
                engine = FrameInferenceEngine(
                    int(SETTINGS.training.max_window_size_seconds * 1000),
                    int(SETTINGS.training.eval_stride_size_seconds * 1000),
                    model,
                    zmuv_transform,
                    # zmuv_transform是零均值单位方差(Zero Mean Unit Variance)变换
                    # 它用于标准化音频特征，确保输入模型的数据分布稳定
                    ctx,
                    # ctx是推理上下文(InferenceContext)
                    # 包含词汇表、标记类型和标签映射等基础配置
                    # zmuv_transform和ctx都是一个与阈值无关的预处理步骤，更换阈值评估时不需要重新加载
                )
            else:
                engine = InferenceEngine(model, zmuv_transform, ctx)
            
            for current_threshold in thresholds:
                engine.set_threshold(float(current_threshold))
                model.eval()
                conf_matrix = ConfusionMatrix()
            
                # 使用线程安全的方式记录错误信息
                if write_errors:
                    with file_lock:
                        with (workspace.path / "errors.tsv").open("a") as error_file:
                            print(prefix, file=error_file)
                            
                pbar = tqdm(dataset, desc=f"{prefix} (阈值={current_threshold})")
                for _, ex in enumerate(pbar):
                    if mixer is not None:
                        (ex,) = mixer([ex])
                    audio_data = ex.audio_data.to(device)
                    engine.reset()
                    seq_present = engine.infer(audio_data)
                    
                    # 使用线程安全方式记录错误
                    if seq_present != positive_set and write_errors:
                        with file_lock:
                            with (workspace.path / "errors.tsv").open("a", encoding='utf-8') as error_file:
                                error_file.write(
                                    f"{ex.metadata.transcription}\t{int(seq_present)}\t{int(positive_set)}\t{ex.metadata.path}\t阈值:{current_threshold}\n"
                                )
                                
                    conf_matrix.increment(seq_present, positive_set)
                    pbar.set_postfix(dict(mcc=f"{conf_matrix.mcc}", c=f"{conf_matrix}"))

                Logger.info(f"{prefix} @ {engine.threshold}: {conf_matrix}")
                Logger.info(f"在阈值 {engine.threshold}，推理序列: {engine.sequence}下的详细评估结果:")
                Logger.info(f"真阳性(TP): {conf_matrix.tp}, 真阴性(TN): {conf_matrix.tn}")
                Logger.info(f"假阳性(FP): {conf_matrix.fp}, 假阴性(FN): {conf_matrix.fn}")
            
                # 线程安全方式保存和记录结果
                if save and not args.eval and positive_set:
                    with file_lock:
                        writer.add_scalar(f"{prefix}/Metric/tp_rate", conf_matrix.tp / len(dataset), epoch_idx)
                        workspace.increment_model(model, conf_matrix.tp)
                
                if args.eval:
                    threshold = engine.threshold
                    with file_lock:
                        with (workspace.path / (str(round(threshold, 2)) + "_results.csv")).open("a", encoding='utf-8') as result_file:
                            result_file.write(
                                f"{prefix},{threshold},{conf_matrix.tp},{conf_matrix.tn},{conf_matrix.fp},{conf_matrix.fn}\n"
                            )

            Logger.info(f"[线程] 完成评估数据集: {prefix}")
            return True
            
        except Exception as e:
            Logger.error(f"[线程] 评估数据集 {prefix} 时出错: {str(e)}")
            raise

    # 在Python 3.6中创建一个简单的nullcontext (Python 3.7+内置了这个)(已经移动至全局作用域)
    @contextlib.contextmanager
    def nullcontext():
        yield

    # deprecated 单线程串行评估
    # def do_evaluate():
    #     """Run evaluation on different datasets"""
    #     evaluate_engine(ww_dev_pos_ds, "Dev positive", positive_set=True)
    #     evaluate_engine(ww_dev_neg_ds, "Dev negative", positive_set=False)
    #     if SETTINGS.training.use_noise_dataset:
    #         evaluate_engine(ww_dev_pos_ds, "Dev noisy positive", positive_set=True, mixer=dev_mixer)
    #         evaluate_engine(ww_dev_neg_ds, "Dev noisy negative", positive_set=False, mixer=dev_mixer)
    #     evaluate_engine(ww_test_pos_ds, "Test positive", positive_set=True)
    #     evaluate_engine(ww_test_neg_ds, "Test negative", positive_set=False)
    #     if SETTINGS.training.use_noise_dataset:
    #         evaluate_engine(
    #             ww_test_pos_ds, "Test noisy positive", positive_set=True, mixer=test_mixer,
    #         )
    #         evaluate_engine(
    #             ww_test_neg_ds, "Test noisy negative", positive_set=False, mixer=test_mixer,
    #         )

    # 使用多线程并行评估
    def do_evaluate():
        """Run evaluation on different datasets in parallel"""
        Logger.heading("====== 开始并行评估模型 ======")
        
        # 创建文件锁，用于同步文件访问，主要是threshold_results.csv文件的写入
        file_lock = threading.Lock()
        # 该所被用于控制多个线程同时写入文件，避免文件写入冲突
        # 涉及文件包括错误日志文件errors.tsv和阈值结果文件threshold_results.csv
        # 在保存模型和记录评估指标时，也会使用文件锁
        # if save and not args.eval and positive_set:
        #     with file_lock:
        #         writer.add_scalar(f"{prefix}/Metric/tp_rate", conf_matrix.tp / len(dataset), epoch_idx)
        #         workspace.increment_model(model, conf_matrix.tp)
        
        
        # 准备评估任务列表
        # TODO: to avoid thread conflicts, disable writing errors for all datasets
        eval_tasks = [
            # 参数分别对应dataset, prefix, positive_set, save, write_errors, mixer
            (ww_dev_pos_ds, "Dev positive", True, False, True, None),
            (ww_dev_neg_ds, "Dev negative", False, False, True, None),
            (ww_test_pos_ds, "Test positive", True, False, True, None),
            (ww_test_neg_ds, "Test negative", False, False, True, None),
        ]
        
        if SETTINGS.training.use_noise_dataset:
            eval_tasks.extend([
                # 参数分别对应dataset, prefix, positive_set, save, write_errors, mixer
                (ww_dev_pos_ds, "Dev noisy positive", True, False, True, dev_mixer),
                (ww_dev_neg_ds, "Dev noisy negative", False, False, True, dev_mixer),
                (ww_test_pos_ds, "Test noisy positive", True, False, True, test_mixer),
                (ww_test_neg_ds, "Test noisy negative", False, False, True, test_mixer),
            ])
        
        # 并行执行评估任务
        # 确定线程池大小：最大线程数为评估任务数量
        max_workers = len(eval_tasks)
        Logger.info(f"启动并行评估，使用{max_workers}个线程处理{len(eval_tasks)}个评估任务")
        
        # TODO: 尝试使用ProcessPoolExecutor，注意监测内存占用情况，避免内存泄漏
        # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_task = {
                executor.submit(
                    evaluate_engine_threaded, 
                    dataset, prefix, positive_set, save, write_errors, mixer, file_lock
                ): prefix 
                for dataset, prefix, positive_set, save, write_errors, mixer in eval_tasks
            }
            
            # 等待所有任务完成并处理结果
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    future.result()  # 获取结果，如有异常会重新抛出
                    Logger.info(f"任务 '{task_name}' 完成")
                except Exception as exc:
                    Logger.error(f"任务 '{task_name}' 执行失败: {exc}")
                    # 记录异常的详细堆栈跟踪
                    import traceback
                    Logger.error(traceback.format_exc())
        
        Logger.heading("====== 所有评估线程已运行完成 ======")

    if args.eval:
        Logger.heading("Model evaluation")
        Logger.info("=============== Evaluation Configuration ===============")
        Logger.info(f"Model: {args.model}")
        Logger.info(f"Device: {device}")
        Logger.info(f"Dataset size - Train: {len(ww_train_ds)}, Dev: {len(ww_dev_ds)}, Test: {len(ww_test_ds)}")
        Logger.info("================================================")
        workspace.load_model(model, best=not args.load_last)
        # 关于--load-weights参数的说明：在指定了--eval（仅进行[评估]时），不需要单独指定--load-weights参数，会默认自动加载模型权重
        Logger.info(SETTINGS)        
        Logger.info("=============== Evaluate Started ===============")
        do_evaluate()
        Logger.info("=============== Evaluate Completed ===============")
        # 如果设置为--eval，则测试后直接返回，不进行训练等操作
        return

    # region train model
    Logger.heading("Model training")
    Logger.info("=============== Training Configuration ===============")
    Logger.info(f"Model: {args.model}")
    Logger.info(f"Number of epochs: {SETTINGS.training.num_epochs}")
    Logger.info(f"Batch size: {SETTINGS.training.batch_size}")
    Logger.info(f"Learning rate: {SETTINGS.training.learning_rate}")
    Logger.info(f"Weight decay: {SETTINGS.training.weight_decay}")
    Logger.info(f"Device: {device}")
    Logger.info(f"Dataset size - Train: {len(ww_train_ds)}, Dev: {len(ww_dev_ds)}, Test: {len(ww_test_ds)}")
    Logger.info("================================================")
    workspace.write_args(args)
    workspace.save_settings(SETTINGS)
    Logger.info(SETTINGS)
    writer.add_scalar("Meta/Parameters", sum(p.numel() for p in params))

    spectrogram_augmentations = (SpecAugmentTransform().train(),)
    spectrogram_augmentations = compose(*spectrogram_augmentations)

    Logger.info("=============== Training Started ===============")
    pbar = trange(SETTINGS.training.num_epochs, position=0, desc="Training", leave=True)
    for epoch_idx in pbar:
        Logger.info(f"Epoch {epoch_idx}, lr: {optimizer.param_groups[0]['lr']}, batch_size: {SETTINGS.training.batch_size}, device: {device}, training...")
        model.train()
        audio_transform.train()
        model.streaming_state = None
        total_loss = torch.Tensor([0.0]).to(device)
        for batch in train_dl:
            batch.to(device)
            audio_length = audio_transform.compute_lengths(batch.lengths)
            zmuv_audio_data = zmuv_transform(audio_transform(batch.audio_data))
            augmented_audio_data = spectrogram_augmentations(zmuv_audio_data)
            
            if use_frame:
                scores = model(augmented_audio_data, audio_length)
                loss = criterion(scores, batch.labels)
            else:
                scores = model(augmented_audio_data, audio_length)
                scores = F.log_softmax(scores, -1)  # [num_frames x batch_size x num_labels]
                audio_length = torch.tensor([model.compute_length(x.item()) for x in audio_length]).to(device)
                loss = criterion(scores, batch.labels, audio_length, batch.label_lengths)
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss

        for group in optimizer.param_groups:
            group["lr"] *= SETTINGS.training.lr_decay

        mean_loss = total_loss / len(train_dl)
        pbar.set_postfix(dict(loss=f"{mean_loss.item():.3}"))
        Logger.info(f"Epoch {epoch_idx} 完成，平均损失: {mean_loss.item():.6f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")

        writer.add_scalar("Training/Loss", mean_loss.item(), epoch_idx)
        # TODO: group["lr"] is invalid
        # pylint: disable=undefined-loop-variable
        writer.add_scalar("Training/LearningRate", group["lr"], epoch_idx)

        if epoch_idx % args.eval_freq == 0 and epoch_idx != 0:
            Logger.info("\nRunning validation...")
            evaluate_engine(
                ww_dev_pos_ds, "Dev positive", positive_set=True, save=True, write_errors=False,
            )
        # 在训练过程中的验证阶段dev（每隔 eval_freq 个epoch进行一次验证）确实只在 Dev positive 数据集上进行验证：
        # 调用 evaluate_engine 函数，设置save=True，表示在验证过程中保存模型和记录指标。
        # 由于 save=True，会在验证过程中保存模型: workspace.increment_model(model, conf_matrix.tp) 并记录指标: writer.add_scalar
        # 如果当前模型的真阳性(TP)指标比之前保存的模型更好，就会更新保存的"最佳模型"。用于在训练过程中保存最佳模型model-best.pt.bin

        # evaluate_engine 能够访问 epoch_idx 变量，这是由 Python 的**闭包(closure)**特性实现的。
        # Python 闭包机制
        # evaluate_engine 函数定义在 main() 函数内部，这使它成为一个嵌套函数。Python 的闭包机制允许嵌套函数访问其外部作用域中定义的变量，即使在调用时这些变量不是显式传递的参数。

    # 如果不指定--eval参数，结束训练后直接退出，不进行模型评估，并打印一边模型信息
    
    Logger.info("Model training completed.")
    Logger.info("Model information:")
    Logger.info(model)
    Logger.info("Training completed.")
    Logger.info("=============== Training Completed ===============")
    # Logger.heading("Model evaluation")
    # do_evaluate()
    # endregion train model


if __name__ == "__main__":
    main()
