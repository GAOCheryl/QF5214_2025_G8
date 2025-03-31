FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including gcc
RUN apt-get update && apt-get -y install cron gcc python3-dev

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script
COPY final_sentiment_strategy.py .

# Create crontab file
RUN echo "0 13 * * * cd /app && python /app/final_sentiment_strategy.py >> /var/log/cron.log 2>&1" > /etc/cron.d/script-cron
RUN chmod 0644 /etc/cron.d/script-cron

# Apply cron job
RUN crontab /etc/cron.d/script-cron

# Create log file
RUN touch /var/log/cron.log

# Run cron in foreground with JSON format
CMD ["cron", "-f"]