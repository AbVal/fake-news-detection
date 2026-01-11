FROM pytorch/torchserve:latest

RUN mkdir -p model_store

RUN pip install --no-cache-dir transformers==4.57.1

COPY torchserve/custom_handler.py .

COPY model_store/distilbert.mar model_store/

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["torchserve", \
     "--start", \
     "--disable-token-auth", \
     "--model-store", "model_store", \
     "--models", "distilbert=distilbert.mar"]
