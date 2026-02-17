FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY space_requirements.txt /tmp/space_requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/space_requirements.txt

COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
