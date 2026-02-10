FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python", "manager.py"]
