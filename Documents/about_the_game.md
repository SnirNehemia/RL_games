# About The Snake Game

This project implements the classic game of Snake, where a machine learning agent is trained to play.

![Snake Game Screenshot](images/game_screenshot.png)
*(To see this image, run the game in human mode and take a screenshot. Save it as `game_screenshot.png` inside an `images` subfolder within `Documents`.)*

## Gameplay

The objective of the game is for the snake (blue) to navigate the board and eat the food (red).

-   The snake is controlled by an AI agent.
-   The agent can choose one of three actions at each step: go straight, turn right, or turn left, relative to its current direction.
-   When the snake eats a piece of food, it grows longer, and the score increases.
-   A new piece of food appears at a random location on the board.

## Game Over Conditions

The game ends if any of the following conditions are met:
1.  The snake's head collides with the boundaries of the game window.
2.  The snake's head collides with any part of its own body.
3.  The snake wanders for too long without eating food. This is based on a configurable step limit that scales with the snake's length.

## Reward System

The AI agent is trained using a reward system to encourage desired behavior:

-   **`+10` points:** For successfully eating a piece of food.
-   **`-10` points:** As a penalty for losing the game (any game-over condition).
-   **`0` points:** For any other move that does not result in eating food or losing the game.

This reward structure incentivizes the agent to seek food while actively avoiding collisions and inefficient paths.