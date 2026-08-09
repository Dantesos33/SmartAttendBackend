# Lock the base image to stable Debian Bookworm to ensure package availability
FROM python:3.10-slim-bookworm

# Install modern system packages required for compiling dlib and running OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install your project python packages smoothly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend files
COPY . .

# Start FastAPI on Railway's dynamic port configuration
ENV PYTHONUNBUFFERED=1

# Apply schema changes before serving traffic. Existing create_all-based
# databases are adopted by migrate_db.py without recreating their tables.
CMD ["sh", "-c", "python migrate_db.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
