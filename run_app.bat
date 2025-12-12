@echo off
REM Use the python on PATH — change full path if you want explicit interpreter
REM Prevent transformers from loading TensorFlow
set TRANSFORMERS_NO_TF=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM Start the app
python -m streamlit run Resume_Screening_Bot_App.py --server.port 8501
pause
