# Steam Game Metacritic Prediction

Steam Game Metacritic Prediction is a collection of data-pipeline, feature-extraction, and prediction with ML/DL models.
- Data collection with Web Crawling and API
- Data cleaning
- Feature extraction with Neural network
- Prediction with ensenble tree-models
- Prediction with Neural network

##
The gaming market is huge

## Workflow
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Setup
1) Download code
```bash
git clone ....
cd steam-git
```
2) Set up a virtual environment with required libraries.
```bash
python -m venv steam-env
source steam-env/bin/activate
pip install requirements.txt
```
3) Prepare data 
You can download the data from gdrive here, or you can collect it manually.
```bash
# crawl tags from steam website
python main.py crawl -s tag 

# get data from steam api
python main.py crawl -s data
```

## Usage

data.yaml
```yaml
- images: list of image file paths
- description: string of game description
- 
```

1) Get features

You need yaml file with raw data
```bash
python main.py extract -d demo_data.yaml -f all
```

```bash
python main.py extract -f graphic
```

Available features</br>
- graphic: 2D, 3D, Cartoon, Pixel, Realistic, Top-Down, Isometric, First-person, Third-person, Resolution
- contents: Story-rich, Open world, Choices Matter, Multiple Endings
- genre:
- theme:
- mood:
- mechanism: Fight, Shoot, Combat, Platformer, Hack-and-Slash, Survive, Build-and-Create,
- players: Single, Multi_local, Multi_online, Competitve, Co-op

- get images and text dataloader
features


3) Predict metacritic score
```bash
python main.py predict -d demo_data.yaml
```
demo_data then you can see the score.
