import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import random
import pickle
import os
import matplotlib.pyplot as plt

# === Q-learning Agent Class ===
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, file_path="q_table.pkl"):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.file_path = file_path
        self.load_q_table()

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state, next_actions):
        max_q_next = max([self.get_q(next_state, a) for a in next_actions], default=0.0)
        current_q = self.get_q(state, action)
        self.q[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)

    def save_q_table(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self.q, f)

    def load_q_table(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            try:
                with open(self.file_path, "rb") as f:
                    self.q = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                self.q = {}
        else:
            self.q = {}

    def clear_q_table(self):
        self.q = {}
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# === Utility Functions ===
def minimax(board, player):
    winner = check_winner(board)
    if winner == 2:
        return 1, None
    elif winner == 1:
        return -1, None
    elif is_full(board):
        return 0, None

    best_score = -float('inf') if player == 2 else float('inf')
    best_move = None
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                board[r][c] = player
                score, _ = minimax(board, 3 - player)
                board[r][c] = 0
                if (player == 2 and score > best_score) or (player == 1 and score < best_score):
                    best_score = score
                    best_move = (r, c)
    return best_score, best_move

def is_full(board):
    return all(cell != 0 for row in board for cell in row)

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def board_to_state(board):
    return tuple(cell for row in board for cell in row)

# === Main GUI Application ===
class RLMinimaxTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning vs Minimax")
        self.agent = QLearningAgent()
        self.board = [[0]*3 for _ in range(3)]
        self.buttons = [[None]*3 for _ in range(3)]
        self.victories_rl = 0
        self.victories_mm = 0
        self.draws = 0
        self.history = []

        for r in range(3):
            for c in range(3):
                btn = tk.Button(root, text="", font=('Arial', 24), width=5, height=2,
                                command=lambda r=r, c=c: self.player_move(r, c))
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

        self.label = tk.Label(root, text=self.get_stats_text(), font=('Arial', 12))
        self.label.grid(row=3, column=0, columnspan=3)

        tk.Button(root, text="Train vs Minimax", command=self.train).grid(row=4, column=0)
        tk.Button(root, text="Reset Q-Table", command=self.reset_qtable).grid(row=4, column=1)
        tk.Button(root, text="Reset Board", command=self.reset_board).grid(row=4, column=2)
        tk.Button(root, text="Plot Performance", command=self.plot_performance).grid(row=5, column=0, columnspan=3)
        tk.Button(root, text="Auto-Play Minimax vs RL", command=self.autoplay_minimax_vs_rl).grid(row=6, column=0, columnspan=3)

    def get_stats_text(self):
        total = self.victories_rl + self.victories_mm + self.draws
        if total == 0:
            return "RL: 0% | Minimax: 0% | Draws: 0%"
        return f"RL: {100 * self.victories_rl / total:.1f}% | Minimax: {100 * self.victories_mm / total:.1f}% | Draws: {100 * self.draws / total:.1f}%"

    def reset_board(self):
        self.board = [[0]*3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="", state="normal")

    def player_move(self, r, c):
        if self.board[r][c] == 0:
            self.board[r][c] = 1
            self.buttons[r][c].config(text="X", state="disabled")
            if check_winner(self.board):
                messagebox.showinfo("Game", "You win!")
                self.reset_board()
            elif is_full(self.board):
                messagebox.showinfo("Game", "Draw!")
                self.reset_board()
            else:
                self.root.after(300, self.ai_move)

    def ai_move(self):
        state = board_to_state(self.board)
        actions = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == 0]
        move = self.agent.choose_action(state, actions)
        self.board[move[0]][move[1]] = 2
        self.buttons[move[0]][move[1]].config(text="O", state="disabled")
        if check_winner(self.board):
            messagebox.showinfo("Game", "RL wins!")
            self.reset_board()
        elif is_full(self.board):
            messagebox.showinfo("Game", "Draw!")
            self.reset_board()

    def train(self):
        n = simpledialog.askinteger("Train vs Minimax", "How many games to train against Minimax?", minvalue=10, initialvalue=500)
        if n is None:
            return
        cancel_window = tk.Toplevel(self.root)
        cancel_window.title("Training Progress")
        tk.Label(cancel_window, text="Training in progress...").pack(pady=10)
        progress = tk.DoubleVar()
        progress_bar = ttk.Progressbar(cancel_window, maximum=n, variable=progress, length=300)
        progress_bar.pack(pady=10)
        cancel = tk.BooleanVar(value=False)
        tk.Button(cancel_window, text="Cancel", command=lambda: cancel.set(True)).pack(pady=5)

        self.root.update()
        for i in range(n):
            if cancel.get():
                cancel_window.destroy()
                messagebox.showinfo("Training Cancelled", f"Training cancelled after {i} games.")
                return

            board = [[0]*3 for _ in range(3)]
            player = 1
            winner = 0
            states_actions = []

            while not winner and not is_full(board):
                state = board_to_state(board)
                actions = [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]
                if player == 2:
                    move = self.agent.choose_action(state, actions)
                else:
                    _, move = minimax(board, player)

                board[move[0]][move[1]] = player
                next_state = board_to_state(board)
                next_actions = [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]
                states_actions.append((state, move, next_state, next_actions, player))
                winner = check_winner(board)
                player = 3 - player

            for s, a, ns, na, p in states_actions:
                if p == 2:
                    reward = 1 if winner == 2 else -1 if winner == 1 else 0
                    self.agent.learn(s, a, reward, ns, na)

            if winner == 2:
                self.victories_rl += 1
            elif winner == 1:
                self.victories_mm += 1
            else:
                self.draws += 1

            self.history.append((self.victories_rl, self.victories_mm, self.draws))
            progress.set(i + 1)
            self.root.update()

        cancel_window.destroy()
        self.agent.save_q_table()
        self.label.config(text=self.get_stats_text())
        messagebox.showinfo("Training Completed", f"Training complete. {n} games against Minimax.")

    def reset_qtable(self):
        self.agent.clear_q_table()
        self.victories_rl = 0
        self.victories_mm = 0
        self.draws = 0
        self.history = []
        self.label.config(text=self.get_stats_text())

    def plot_performance(self):
        if not self.history:
            messagebox.showinfo("No Data", "Train the model first.")
            return
        x = list(range(1, len(self.history)+1))
        rl = [r for r, _, _ in self.history]
        mm = [m for _, m, _ in self.history]
        dr = [d for _, _, d in self.history]
        plt.figure(figsize=(10, 5))
        plt.plot(x, rl, label="RL Wins")
        plt.plot(x, mm, label="Minimax Wins")
        plt.plot(x, dr, label="Draws")
        plt.xlabel("Training Iteration")
        plt.ylabel("Cumulative Count")
        plt.title("Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def autoplay_minimax_vs_rl(self):
        n = simpledialog.askinteger("Auto-Play", "How many games to simulate?", minvalue=10, initialvalue=100)
        if not n:
            return

        board_canvas = tk.Toplevel(self.root)
        board_canvas.title("Auto-Play Minimax vs RL")
        board_labels = [[tk.Label(board_canvas, text="", font=('Courier', 36), width=2) for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                board_labels[r][c].grid(row=r, column=c)

        rl_wins, mm_wins, draws = 0, 0, 0
        self.root.update()

        for i in range(n):
            board = [[0]*3 for _ in range(3)]
            player = 1
            winner = 0

            while not winner and not is_full(board):
                state = board_to_state(board)
                actions = [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]
                move = self.agent.choose_action(state, actions) if player == 2 else minimax(board, player)[1]

                board[move[0]][move[1]] = player
                for r in range(3):
                    for c in range(3):
                        val = board[r][c]
                        board_labels[r][c].config(text="X" if val == 1 else "O" if val == 2 else " ")
                board_canvas.update()
                self.root.after(300)

                winner = check_winner(board)
                player = 3 - player

            if winner == 2:
                rl_wins += 1
            elif winner == 1:
                mm_wins += 1
            else:
                draws += 1

        total = rl_wins + mm_wins + draws
        messagebox.showinfo("Auto-Play Results", f"Games: {total}\nRL Wins: {rl_wins} ({rl_wins/total:.1%})\nMinimax Wins: {mm_wins} ({mm_wins/total:.1%})\nDraws: {draws} ({draws/total:.1%})")
        board_canvas.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = RLMinimaxTrainerApp(root)
    root.mainloop()
