FROM python:3.11-slim

WORKDIR /app
ARG PIP_TRUSTED_HOSTS=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    ${PIP_TRUSTED_HOSTS} \
    "torch==2.10.0+cpu" \
    && pip install --no-cache-dir --default-timeout=120 --retries=10 ${PIP_TRUSTED_HOSTS} -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
