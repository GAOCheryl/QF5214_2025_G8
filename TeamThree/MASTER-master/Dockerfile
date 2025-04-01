FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including gcc and set timezone
RUN apt-get update && apt-get -y install cron gcc python3-dev tzdata
# Set your timezone - replace "America/New_York" with your timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script and ensure it has execution permissions
COPY final_sentiment_strategy.py .
RUN chmod +x final_sentiment_strategy.py

# Create data directories that might be needed
RUN mkdir -p data/Input data/Output model/Output

# Create crontab file with proper environment variables
RUN echo "0 13 * * * cd /app && python /app/final_sentiment_strategy.py >> /var/log/cron.log 2>&1" > /etc/cron.d/script-cron
RUN chmod 0644 /etc/cron.d/script-cron

# Apply cron job
RUN crontab /etc/cron.d/script-cron

# Create log file with appropriate permissions
RUN touch /var/log/cron.log && chmod 0666 /var/log/cron.log

# Start cron and tail logs
CMD cron && tail -f /var/log/cron.log