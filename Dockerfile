FROM python:3.10-slim

WORKDIR /app

# Install only what's missing from your base
COPY requirements.txt .

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY rp_handler.py servingLLM.py ./

CMD ["python3", "-u", "rp_handler.py"]