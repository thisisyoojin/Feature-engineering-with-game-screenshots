# Game Feature Extraction with NN

Game Feature Extraction is predicting game features with deep learning model(CNN, RNN) from raw unconstructed data such as text and images.

## Inspiration

As the part of the project of Game Score Prediction, I used tags of steam games for features in the game. Players tag games in the Steam Store with terms that they think are appropriate, or apply tags that others have already suggested for that title. But as it is tagged by users, sometimes incorrect or inappropriate tags are displayed in a game or multiple tags with similar attribute are displayed, which doesn't give useful information.

To solve this issue, I design this project to extract these tags(features) from game descriptions and screenshots. With this process, I can extract necessary features from a game and use it to predict the game's metacritic score in future.


## Project workflow
- Images</br>
With a game's screenshots, models will predict multi-label in themes and graphic features. The models will be a different variaty of multi-label Convolutional Neural Networks.

- Text</br>
With a game's description, models will predict multi-label in contents, mechanism, and player features. The models will be a different variaty of multi-label Convolutional Neural Networks or Recurrent Neural Networks.

## Design
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Output
genre_best_model.pkl
theme_best_model.pkl
mood_best_model.pkl
graphic_best_model.pkl
contents_best_model.pkl
mechanism_best_model.pkl
players_best_model.pkl

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
- mood: Violent, Funny, Horror, Sexual, Relaxing, Dark, Mystery, Atmospheric, Cute
- theme: Sci-fi-Mechs, Post-apocalyptic, Retro, Zombies, Military, Fantasy, Historical
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

