import chess
import torch
import numpy as np
from model.network import ChessNet
from engine.encoder import board_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessNet().to(device)
model.load_state_dict(torch.load("model/chess_model.pt", map_location=device))
model.eval()

def get_neural_move(board, top_k=5):
    tensor = board_to_tensor(board)
    x = torch.tensor(tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(x)
    
    legal_moves = list(board.legal_moves)
    best_move = legal_moves[0]
    best_score = -float('inf')
    
    policy = policy.squeeze().cpu().numpy()
    
    for move in legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        score = policy[to_sq]
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move, float(value.squeeze())

if __name__ == "__main__":
    board = chess.Board()
    move, value = get_neural_move(board)
    print(f"Neural move: {move}")
    print(f"Position value: {value:.3f}")

'''python -m engine.neural_search '''