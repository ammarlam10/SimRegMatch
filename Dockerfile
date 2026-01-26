FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

WORKDIR /workspace

# PyTorch base image already has torch and torchvision installed
# No system dependencies needed - PIL/Pillow works with Python packages alone

# Copy requirements file
COPY requirements.txt .

# Install remaining Python dependencies (torch/torchvision already installed)
# Upgrade numpy first to meet pandas requirements, then install from requirements.txt
RUN pip install --no-cache-dir --upgrade numpy>=1.20.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["/bin/bash"]
