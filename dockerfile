# Use an official Python version that matches your requirements (e.g., 3.9)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

RUN alembic upgrade head

WORKDIR /app/service

CMD uvicorn app:app --host 0.0.0.0