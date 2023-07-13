# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install matplotlib


# Copy the API code into the container
COPY . .

# Expose the port your API will listen on
EXPOSE 3000

# Set the command to run your API
CMD ["python3", "monthly_prediction.py"]
