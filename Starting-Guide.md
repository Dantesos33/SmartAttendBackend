# SmartAttend - Run Guide (Anaconda Prompt)

This guide provides instructions for setting up and running the SmartAttend backend using **Anaconda Prompt** on Windows.

# Requirements

- Python 3.10
- Anaconda
- CMake
- Node.js

# Starting the Backend

## 1. Setting Up the Environment (First Time Only)

Open **Anaconda Prompt Shell** and follow these steps to create a dedicated environment:

```bash
# Navigate to the backend directory
cd "e:\Final Year Project (FYP)\SmartAttend\backend"

# Create a new conda environment with Python 3.10
conda create -n smartattend python=3.10 -y

# Activate the environment
conda activate smartattend

# Install dlib
conda install -c conda-forge dlib

# Install libraries from requirements.txt
pip install -r requirements.txt
```

## 2. Running the Server (Every Time)

```bash
# Navigate to the backend directory
cd "e:\Final Year Project (FYP)\SmartAttend\backend"

# Activate the environment
conda activate smartattend

# Run the Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

# Starting the Frontend

## 1. Setting Up the Environment (First Time Only)

```bash
# Navigate to the frontend directory
cd "e:\Final Year Project (FYP)\SmartAttend\frontend"

# Install dependencies
npm install
npm expo install
```

## 2. Running the Server (Every Time)

```bash
# Run the server
npm start
```
