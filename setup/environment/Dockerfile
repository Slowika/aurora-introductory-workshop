# minimal Dockerfile for the workshop environment
FROM python:3.13-slim

WORKDIR /app
RUN pip install --upgrade pip uv

COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system .