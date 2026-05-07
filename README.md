# LiteRT-LM
A simple way to set up OpenAI compatible server for model inferencing with LiteRT-LM.

## Quick Start
A server can be set up with
```
git clone https://github.com/AnaOnTram/LiteRT-LM.git
cd LiteRT-LM
```
Install the dependencies
```
pip install -r requirements.txt
```
Start the server
```
python server.py --hf litert-community/gemma-4-E2B-it-litert-lm --gpu --context 30000
```
