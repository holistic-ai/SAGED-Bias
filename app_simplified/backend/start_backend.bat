@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --reload-dir ../../saged --port 8000 