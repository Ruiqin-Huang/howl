#bin/bash
# set -e意味着如果脚本中的任何命令以非零状态退出（表示失败），脚本将立即退出. 
# 这是一种错误处理机制，可以防止脚本在发生错误后继续执行，从而避免潜在的问题.
set -e

# wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
# 若ipv4连接失败，使用如下方法强制使用ipv6下载。如果主机 DNS 记录没有 IPv6 地址，wget 会报错 :
# wget --inet6-only https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
tar -xf montreal-forced-aligner_linux.tar.gz
rm montreal-forced-aligner_linux.tar.gz

pushd montreal-forced-aligner

# known mfa issue https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/109
cp lib/libpython3.6m.so.1.0 lib/libpython3.6m.so

# download english dictionary from http://www.openslr.org/11/
# wget https://www.openslr.org/resources/11/librispeech-lexicon.txt

# make sure executable is working
bin/mfa_align -h >/dev/null

popd

# cache folder mfa uses to store temporary alignments
mkdir -p ~/Documents/MFA
