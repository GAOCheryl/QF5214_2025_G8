# Use official Python image
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy only Team Four's Streamlit app files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "StreamlitAPP/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
