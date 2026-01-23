# minimal Dockerfile for the workshop environment
FROM python:3.10.18

# aurora core dependencies, all that's needed to simply run aurora
RUN pip install microsoft-aurora

# optional extras for data manipulation and visualization
RUN pip install xarray zarr dask matplotlib cartopy

# azure ml extras for workspace interaction, auth, and logging
RUN pip install azure-ai-ml mlflow azureml-mlflow