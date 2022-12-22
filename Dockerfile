FROM python:3.9

 # Copy all project files int '/app' inside image  (create it if not exist)
 COPY . /app

# Set the working directory
WORKDIR /app

# Install all dependencies
RUN pip install -r requirements.txt

# Expose the port for the application
EXPOSE 8080

# Run the FastAPI application
CMD ["python","app.py"]
