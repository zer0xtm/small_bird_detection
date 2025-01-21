FROM python:3.10.12-slim

WORKDIR /workspace

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "script.py"]
