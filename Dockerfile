FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pytest-json-report for JSON output
RUN pip install pytest-json-report

# Copy the test script
COPY storage/test_script.py ./test_script.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTEST_CURRENT_TEST=""

# Default command to run pytest with JSON report
CMD ["python", "-m", "pytest", "test_script.py", "--json-report", "--json-report-file=report.json", "-v"]
