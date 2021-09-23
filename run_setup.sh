env/bin/python -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
env/bin/python -m pip install -r requirements.txt
env/bin/python -m pip install spacy
env/bin/python -m spacy download en_core_web_sm
:: env/bin/python -m spacy download en_core_web_trf
env/bin/python -m pip install -e ailignment
env/bin/python -m textblob.download_corpora

