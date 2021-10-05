# Steam Game Metacritic Score Prediction

Steam Game Score Prediction is a collection of data-pipeline, feature-extraction, and prediction with ML/DL models. 

## Business value
The gaming market is huge

## Workflow
- Data collection with Web Crawling and API
- Data cleaning & store in AWS
- Feature extraction with Neural network
- Prediction with ensenble tree-models
- Prediction with Neural network

## Design
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

You can get features from game data and predict metacritic score with this code.
You need yaml file with raw data
data.yaml
```yaml
- images: list of image file paths
- description: string of game description
- 
```

1) Get features
Features
```bash
python main.py extract -d demo_data.yaml
```
Return example is like this:
```bash
This game's features are:
2D, Sci-fi, Pixel, Story-rich, Fight, Single player
```
2) Features probability

Available features</br>
- genre: Action, Adventure, Casual, Puzzle, RPG, Simulation, Strategy, Racing, Arcade, Sports
- theme: Sci-fi-Mechs, Post-apocalyptic, Retro, Zombies, Military, Fantasy, Historical
- mood: Violent, Funny, Horror, Sexual,
- graphic: 2D, 3D, Cartoon, Pixel, Realistic, Top-Down, Isometric, First-person, Third-person, Resolution
- contents: Story-rich, Open world, Choices Matter, Multiple Endings
- mechanism: Fight, Shoot, Combat, Platformer, Hack-and-Slash, Survive, Build-and-Create,
- players: Single, Multi_local, Multi_online, Competitve, Co-op

```bash
python main.py extract -f graphic
```
Return example is like this:
```bash
This game's features are:
2D, Sci-fi, Pixel, Story-rich, Fight, Single player
```

2) Predict metacritic score
If you want to predict predict metacritic score.

```bash
python main.py score -d demo_data.yaml
```
demo_data then you can see the score.
