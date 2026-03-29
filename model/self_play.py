import chess
import torch
import numpy as np
from model.network import ChessNet
from engine.mcts import mcts_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def self_play_game(model, simulations=50):
    board = chess.Board()
    states, policies, turns = [], [], []

    while not board.is_game_over() and len(states) < 100:
        tensor = np.zeros((119, 8, 8), dtype=np.float32)
        from engine.encoder import board_to_tensor
        tensor = board_to_tensor(board)
        states.append(tensor)
        turns.append(board.turn)

        move = mcts_search(board, model, device, simulations=simulations)
        policy = np.zeros(64, dtype=np.float32)
        policy[move.to_square] = 1.0
        policies.append(policy)
        board.push(move)

    result = board.result()
    if result == "1-0": z = 1.0
    elif result == "0-1": z = -1.0
    else: z = 0.0

    values = []
    for turn in turns:
        if turn == chess.WHITE:
            values.append(z)
        else:
            values.append(-z)

    return states, policies, values

def train_self_play(num_games=20, epochs=5, simulations=50):
    model = ChessNet().to(device)
    try:
        model.load_state_dict(torch.load("model/chess_model.pt", map_location=device))
        print("Loaded existing model!")
    except:
        print("Starting fresh model!")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    value_loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        all_states, all_values = [], []

        print(f"\nEpoch {epoch+1}/{epochs} — Generating {num_games} self-play games...")
        for g in range(num_games):
            states, _, values = self_play_game(model, simulations)
            all_states.extend(states)
            all_values.extend(values)
            if (g+1) % 5 == 0:
                print(f"  Game {g+1}/{num_games} done")

        x = torch.tensor(np.array(all_states)).to(device)
        v = torch.tensor(np.array(all_values), dtype=torch.float32).to(device)

        dataset = torch.utils.data.TensorDataset(x, v)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        total_loss = 0
        for xb, vb in loader:
            optimizer.zero_grad()
            _, pred = model(xb)
            loss = value_loss_fn(pred.squeeze(), vb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "model/chess_model.pt")
        print("Model saved!")

if __name__ == "__main__":
    print("Starting RL self-play training...")
    print("Device:", device)
    train_self_play(num_games=20, epochs=5, simulations=50)


'''python -m model.self_play'''