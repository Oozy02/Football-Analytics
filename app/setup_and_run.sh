#!/bin/bash

# Log file to capture all output
LOGFILE="setup_and_run.log"
echo "Starting setup and running the Streamlit app" > $LOGFILE

# Detect the operating system
OS=$(uname -s)

# Step 1: Create a virtual environment
echo "Creating virtual environment..." | tee -a $LOGFILE
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    python3 -m venv venv
elif [[ "$OS" == "MINGW32_NT"* || "$OS" == "MINGW64_NT"* || "$OS" == "CYGWIN"* ]]; then
    python -m venv venv
else
    echo "Unsupported OS: $OS" | tee -a $LOGFILE
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Error creating virtual environment" | tee -a $LOGFILE
    exit 1
else
    echo "Virtual environment created successfully." | tee -a $LOGFILE
fi

# Step 2: Activate the virtual environment
echo "Activating virtual environment..." | tee -a $LOGFILE
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    source venv/bin/activate
elif [[ "$OS" == "MINGW32_NT"* || "$OS" == "MINGW64_NT"* || "$OS" == "CYGWIN"* ]]; then
    # Windows uses a different script to activate the virtual environment
    source venv/Scripts/activate
else
    echo "Unsupported OS: $OS" | tee -a $LOGFILE
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Error activating virtual environment" | tee -a $LOGFILE
    exit 1
else
    echo "Virtual environment activated successfully." | tee -a $LOGFILE
fi

# Step 3: Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..." | tee -a $LOGFILE
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing dependencies" | tee -a $LOGFILE
    exit 1
else
    echo "Dependencies installed successfully." | tee -a $LOGFILE
fi

# Step 4: Run the Streamlit app
echo "Running the Streamlit app..." | tee -a $LOGFILE
streamlit run app.py

if [ $? -ne 0 ]; then
    echo "Error running Streamlit app" | tee -a $LOGFILE
    exit 1
else
    echo "Streamlit app is running." | tee -a $LOGFILE
fi
