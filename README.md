# Game Feature Extraction with NN

Game Feature Extraction is predicting game features with deep learning model(CNN, RNN) from raw unconstructed data such as text and images.

## Inspiration
As the part of the project of Game Score Prediction()



## Project workflow
- CNN: Images

- RNN
 
- Genre

## Design
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)


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

