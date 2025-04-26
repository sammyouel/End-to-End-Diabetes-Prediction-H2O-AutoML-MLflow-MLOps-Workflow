# Dockerfile
# Defines the container environment for the H2O Diabetes Prediction project

# --- Base Image ---
    FROM python:3.10-slim

    # --- Environment Variables ---
    WORKDIR /app
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1
    
    # --- System Dependencies ---
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*
    
    # Verify Java installation
    RUN java -version
    
    # --- Python Dependencies ---
    COPY requirements.txt .
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt
    
    # --- Application Code ---
    # Copy everything AFTER dependencies are installed
    COPY . .
    
    # --- Ports (Optional, for API) ---
    EXPOSE 5000
    
    # --- Default Command ---
    # Runs the main training script
    CMD ["python", "predict_diabetes.py"]