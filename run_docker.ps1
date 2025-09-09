# 빌드: PowerShell에서 실행
cd C:/Users/USER/Desktop/tet

docker build -t slm-train-gpu .

docker run --gpus all --memory=32g -it --rm -v ${PWD}:/workspace slm-train-gpu /bin/bash
