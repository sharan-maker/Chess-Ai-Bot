import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
import random
from engine.encoder import board_to_tensor
from model.network import ChessNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_game():
    board = chess.Board()
    states, moves = [], []
    while not board.is_game_over() and len(states) < 80:
        legal = list(board.legal_moves)
        move = random.choice(legal)
        states.append(board_to_tensor(board))
        moves.append(move)
        board.push(move)
    result = board.result()
    if result == "1-0": value = 1.0
    elif result == "0-1": value = -1.0
    else: value = 0.0
    return states, value

def train(epochs=10, games_per_epoch=100):
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    value_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for _ in range(games_per_epoch):
            states, value = generate_random_game()
            if not states:
                continue
            x = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            v = torch.tensor([value] * len(states), dtype=torch.float32).to(device)
            optimizer.zero_grad()
            _, pred_value = model(x)
            loss = value_loss_fn(pred_value.squeeze(), v)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/games_per_epoch:.4f}")

    torch.save(model.state_dict(), "model/chess_model.pt")
    print("Model saved to model/chess_model.pt")

if __name__ == "__main__":
    print("Training on:", device)
    train(epochs=10, games_per_epoch=100)

'''python -m model.train'''