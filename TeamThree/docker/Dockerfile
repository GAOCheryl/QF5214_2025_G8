# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9-slim AS builder

# Set the working directory in the container
WORKDIR /usr/src/app

# Install git to clone the repository
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy the requirements.txt file into the container
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Clone the repository into a temporary directory
RUN git clone https://github.com/SJTU-DMTai/MASTER.git /usr/src/tmp/master

# List the files to confirm cloning
RUN ls -la /usr/src/tmp/master

# Set the working directory to the cloned repository
WORKDIR /usr/src/tmp/master

# Command to run your application
CMD ["python", "main.py"]

# use powershell to run the following commands in windows
# cd docker
# docker build -t qf5214_group8_t3_image -f Dockerfile .
# docker run qf5214_group8_t3_image