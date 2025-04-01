FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including gcc and set timezone
RUN apt-get update && apt-get -y install cron gcc python3-dev tzdata git

# Extra dependencies that might be needed for some Python packages
RUN apt-get install -y --no-install-recommends build-essential libpq-dev

# Set timezone to Beijing time
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Copy requirements.txt and remove qlib entry
COPY requirements.txt /tmp/original_requirements.txt
RUN cat /tmp/original_requirements.txt | grep -v qlib > /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install qlib directly from GitHub
RUN pip install --no-cache-dir git+https://github.com/microsoft/qlib.git@main

# Copy all required Python scripts and ensure execution permissions
COPY final_sentiment_strategy.py .
COPY master_strategy.py .
COPY base_model_strategy.py .
COPY database_utils.py .
# Copy the alpha_101 folder
COPY alpha_101 ./alpha_101/
COPY model ./model/
RUN chmod +x final_sentiment_strategy.py

# Create data directories that might be needed
RUN mkdir -p data/Input data/Output model/Output

# Create a bash script to run your Python script with proper environment
RUN echo '#!/bin/bash' > /app/run_script.sh && \
    echo 'cd /app' >> /app/run_script.sh && \
    echo 'export PYTHONPATH=$PYTHONPATH:/data:/app' >> /app/run_script.sh && \
    echo '/usr/local/bin/python /app/final_sentiment_strategy.py' >> /app/run_script.sh
RUN chmod +x /app/run_script.sh

# Set up cron job with proper environment
RUN (crontab -l 2>/dev/null; echo "0 13 * * * /app/run_script.sh >> /var/log/cron.log 2>&1") | crontab -

# Create log file
RUN touch /var/log/cron.log && chmod 0666 /var/log/cron.log

# Start cron in foreground to keep container running
CMD service cron start && tail -f /var/log/cron.log