.\env\Scripts\python.exe -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
.\env\Scripts\python.exe -m pip install -r requirements.txt
.\env\Scripts\python.exe -m pip install spacy
.\env\Scripts\python.exe -m spacy download en_core_web_sm
:: .\env\Scripts\python.exe -m spacy download en_core_web_trf
.\env\Scripts\python.exe -m pip install -e ailignment
.\env\Scripts\python.exe -m textblob.download_corpora
pause