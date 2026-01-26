FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifact
COPY predict.py model.bin ./

# Expose API port
EXPOSE 9696

# Run with a production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
