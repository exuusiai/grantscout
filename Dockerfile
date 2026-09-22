FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m scout
USER scout
EXPOSE 8000
VOLUME ["/app/data", "/app/runs"]
CMD ["grantscout", "serve", "--host", "0.0.0.0", "--port", "8000"]
