FROM python:3.10

# 작업 디렉토리
WORKDIR /code

# 필요한 파일 복사
COPY requirements.txt .

# 패키지 설치
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# FastAPI 서버 시작
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
