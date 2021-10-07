#/bin/bash

echo "Cleaning up old environment"
rm -rf env

echo "Creating VENV"
python3.7 -m venv env

echo "Activating VENV" 
source env/bin/activate

echo "Upgrading pip"
pip install --upgrade pip

# echo "Installing dependencies"
# pip3 install -r requirements.txt


