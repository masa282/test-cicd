# ベースイメージ
FROM python:3.9-slim

WORKDIR /app

COPY task.py /app
COPY requirements.txt /app


# 必要なPythonライブラリをインストール
RUN pip install -r requirements.txt --no-cache-dir


# コンテナが起動したときに実行するコマンド
#ENTRYPOINT ["python", "task.py"]
