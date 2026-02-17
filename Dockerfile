FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY space_requirements.txt /tmp/space_requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/space_requirements.txt \
    && python -c "import gradio, requests; print('deps-ok')"

COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
