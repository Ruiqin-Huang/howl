#bin/bash
set -e
set -v

COMMON_VOICE_DATASET_PATH=${1} # common voice dataset path
DATASET_NAME=${2} # underscore separated wakeword (e.g. hey_fire_fox)
INFERENCE_SEQUENCE=${3} # inference sequence (e.g. [0,1,2])
# ${4} pass true to skip generating negative dataset

# 如果参数少于3个，那么输出错误信息，然后退出
if [ $# -lt 3 ]; then
    printf 1>&2 "invalid arguments: ./generate_dataset.sh <common voice dataset path> <underscore separated wakeword> <inference sequence>"
    exit 2
# 如果参数等于4个，那么将SKIP_NEG_DATASET设置为第四个参数
elif [ $# -eq 4 ]; then
    SKIP_NEG_DATASET=${4}
else
# 否则（参数等于3个），将SKIP_NEG_DATASET设置为false
    SKIP_NEG_DATASET="false"
fi

# 输出COMMON_VOICE_DATASET_PATH、DATASET_NAME和INFERENCE_SEQUENCE
printf "COMMON_VOICE_DATASET_PATH: ${COMMON_VOICE_DATASET_PATH}\n"
printf "DATASET_NAME: ${DATASET_NAME}\n"
printf "INFERENCE_SEQUENCE: ${INFERENCE_SEQUENCE}\n"