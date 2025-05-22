# Use official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:churn-api", "--host", "0.0.0.0", "--port", "8005"]