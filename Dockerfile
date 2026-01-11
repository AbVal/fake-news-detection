# preparation
FROM python:3.10-slim AS builder

WORKDIR /project

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update

COPY requirements_docker.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /project/wheels torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /project/wheels -r requirements_docker.txt

# final stage
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /project

COPY --from=builder /project/wheels /wheels
COPY --from=builder /project/requirements_docker.txt .

RUN pip install --no-cache /wheels/*

COPY src/ ./src/

COPY model/ ./model/

ENTRYPOINT ["python3", "-m", "src.predict"]
