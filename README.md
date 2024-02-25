# Part Of Speech Tagging

## Usage

Poetry is needed to run this project. 
To change the dataset and model among other things, go to python/ner/config.py 
Steps to follow
1. Clone the project
2. Go to the python folder and do poetry lock -> poetry install
3. Run cli.py train <output_directory>

Alternatively you can build a docker container from the provided dockerfile and run the code on the container.

## Model

DeBERTa v3 large model is used in this project, however you can change the model from config.py. 

## Dataset

The POS tagging model is trained on the conll2003 dataset. The repository could be used with any Huggingface POS dataset with minimal changes.
The Token labels can be found in /python/pos_tagging/data/pos_tags.py.

