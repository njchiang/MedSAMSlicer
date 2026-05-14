# Use the base CUDA image
# FROM nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:26.03-py3 

# Install Python 3 and system dependencies
# RUN apt-get update && apt-get install -y \
#     python3 python3-pip
#     libgomp1 git \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files needed for package installation (sam2/ must be copied before pip install -e .)
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .
COPY README.md .
COPY sam2/ ./sam2/

# Install PyTorch with CUDA 12.4 support
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install the SAM2 package (builds the CUDA extension)
RUN pip3 install -e .

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/video/segs_tiny

# Expose the Flask port
EXPOSE 8080

# Run the server (default command)
# CMD ["python3", "server.py"]
CMD ["python3", "server.py"]

