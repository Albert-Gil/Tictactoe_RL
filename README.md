
# 🧠 Tic-Tac-Toe: Minimax vs Reinforcement Learning

This project implements a Tic-Tac-Toe game in Python where a **Reinforcement Learning (Q-Learning) agent** competes against a **Minimax algorithm**. It’s based on [Sentdex’s tutorial](https://pythonprogramming.net/) and expanded to allow:

- Head-to-head matches between agents
- Training and evaluation of the Q-Learning model
- Optional human vs AI gameplay

## 🚀 Features

- Q-Learning agent trained from scratch
- Minimax opponent with perfect play
- Configurable training episodes
- Win/draw/loss performance tracking
- Play-by-play printouts for auto-play mode

## 📁 File

- `tic-tac-toe-minimax-vs-RL.py` – main script with everything in one place: training, gameplay, and visualization

## 🎯 Project Goal

This project is intended as a simple, educational case to explore how reinforcement learning (Q-learning) can be implemented.
It is **not** optimized to create a perfect AI player — the goal is to demonstrate how to build and train a basic RL agent using a simple game like Tic-Tac-Toe.

## 🧠 How It Works

- The **Q-Learning agent** learns from trial and error, improving its strategy over thousands of games.
- The **Minimax player** evaluates all possible future states to make the optimal move.
- You can pit them against each other, play yourself, or watch training performance evolve.

## ▶️ Usage

Run the script using:

```bash
python tic-tac-toe-minimax-vs-RL.py
```

You'll be prompted to choose:
- Which agent to train or test
- Number of training games
- Whether to play yourself or watch autoplay

## 📊 Example Output

```
Training in progress...
Episode 1000 | Win rate: 85% | Draw rate: 10% | Loss rate: 5%
```

## 📚 Inspired By

Tutorial series by [Sentdex](https://pythonprogramming.net/), which provides excellent resources on Python and AI.

## 📄 License

MIT License — feel free to use, modify, and share!
