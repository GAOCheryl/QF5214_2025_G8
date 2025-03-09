# Use Python 3.12 base image
FROM python:3.12-slim

# Install dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-distutils \
    python3-setuptools \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory in the container
WORKDIR /getX

# Copy the requirements.txt file into the container
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt

# Copy only your scraper script
COPY get_X_text.py .

# By default, run the script
CMD ["python", "get_X_text.py"]

# use powershell to run the following commands in windows to build
# cd C:\Users\Mr.river\Desktop\getX
# docker build -t qf5214_group8_t1_getX_image:latest .
