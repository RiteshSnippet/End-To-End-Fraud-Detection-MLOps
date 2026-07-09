import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

package="fraud_detection" 

list_of_files = [
    ".github/workflows",
    "notebooks/Data",
    "notebooks/exploratory_data_analysis.ipynb",
    "notebooks/model_training.ipynb",
    f"src/{package}/__init__.py",
    f"src/{package}/exception.py",
    f"src/{package}/logger.py",
    f"src/{package}/utils/__init__.py",
    f"src/{package}/utils/utils.py",
    f"src/{package}/components/__init__.py",
    f"src/{package}/components/data_ingestion.py",
    f"src/{package}/components/data_transformation.py",
    f"src/{package}/components/model_trainer.py",
    f"src/{package}/pipeline/__init__.py",
    f"src/{package}/pipeline/prediction_pipeline.py",
    f"src/{package}/pipeline/training_pipeline.py",
    "static/styles.css",
    "templates/home.html",
    ".gitignore",
    "app.py",
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")