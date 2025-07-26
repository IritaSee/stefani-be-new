FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Clear pip cache and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV PORT=5001
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5001

CMD ["python", "app.py"]
