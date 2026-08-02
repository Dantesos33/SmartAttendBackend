# Lock the base image to stable Debian Bullseye to ensure package availability
FROM python:3.10-slim-bullseye

# Install standard dependencies (Notice libgl1-mesa-glx works perfectly here)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# CRITICAL OPTIMIZATION: Install a pre-compiled dlib wheel directly to bypass a 20-minute compilation
RUN pip install --no-cache-dir https://github.com

# Install the rest of your packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

# Start FastAPI on Railway's port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
