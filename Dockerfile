FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY data/california_housing.csv ./data/
# Model artifacts are not committed; training runs at image build.
RUN python scripts/train.py
EXPOSE 8101
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8101/readyz')"
CMD ["python", "-m", "app.main"]
