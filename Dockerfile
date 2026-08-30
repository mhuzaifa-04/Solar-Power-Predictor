FROM python:3.10-slim

WORKDIR /app

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Clean package manager cache and install essential build tools cleanly
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Container health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Execute Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]