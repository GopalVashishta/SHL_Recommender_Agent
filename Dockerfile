FROM python:3.12-slim

# Create a non-root user (UID 1000) for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies as the non-root user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application and data with correct ownership
COPY --chown=user app ./app
COPY --chown=user data ./data

# HF Spaces expects port 7860
EXPOSE 7860

# Start Uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]