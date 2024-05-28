遇到重要的library，都要新建个环境
python -m venv env-name
source env-name/bin/activate
pip install jupyter lightgbm mljar-supervised
jupyter notebook

查看env：conda env list

激活环境：conda activate lightgbm_env

用conda创建：conda create --name3 myenv python=3.10
用conda删除：conda remove --name env_name --all


——————————————————————下载包————————————————————————————————
conda activate newenv
conda install -c conda-forge package_name
也可能会出错：
    使用 conda install 出错的原因可能有以下几点：
    包未被收录在当前通道：当前你使用的 conda 通道中可能没有包含 h2o 包。默认情况下，conda 使用 Anaconda 和 conda-forge 通道，但某些包可能未被这些通道收录。
    架构限制：你正在使用 macOS ARM 架构（Apple Silicon/M1），而某些包可能尚未针对这个架构进行编译和发布。

