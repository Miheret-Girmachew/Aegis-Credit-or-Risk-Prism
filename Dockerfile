# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set environment variables
# This prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1
# This ensures that Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# 5. Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
# --upgrade ensures we have the latest pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 6. Copy the rest of the application's source code into the container
# This copies the entire 'src' directory
COPY ./src /app/src

# 7. Make port 8000 available to the world outside this container
EXPOSE 8000

# 8. Define the command to run your app using uvicorn
# This command tells uvicorn to look for an object named 'app'
# in the file 'src/api/main.py'.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]