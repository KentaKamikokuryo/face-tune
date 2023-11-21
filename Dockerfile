# Python 3.9のAlpineベースイメージを使用
FROM python:3.9-slim

# タイムゾーン設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libopencv-dev

# PyTorchのインストール
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Pythonの依存関係のインストール
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# アプリケーションのコードをコピー
# 作業ディレクトリの設定
WORKDIR /usr/src/app
COPY . .

# コンテナ起動時のコマンド
CMD ["flask", "run"] 