FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Create the Conda environment based on the environment.yml file
RUN conda env create -f environment.yml

# Activate the environment and install additional dependencies if necessary
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set the environment to activate automatically
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Copy the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
