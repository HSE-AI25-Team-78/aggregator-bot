FROM python:3.11-slim

# Install uv permanently in the image
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency management files
COPY pyproject.toml uv.lock ./

# Install dependencies using the lockfile
RUN uv sync --locked --no-cache

# Copy application code (after dependencies for better caching)
COPY . .

# Run database migrations
RUN uv run alembic upgrade head

WORKDIR /app/service

CMD uv run uvicorn app:app --host 0.0.0.0