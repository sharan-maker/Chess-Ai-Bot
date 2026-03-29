import chess.pgn
import numpy as np
import os
from engine.encoder import board_to_tensor

def parse_pgn(pgn_file, max_games=10000):
    states = []
    values = []
    
    print(f"Parsing {pgn_file}...")
    with open(pgn_file, encoding="utf-8", errors="ignore") as f:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            result = game.headers.get("Result", "*")
            if result == "1-0": value = 1.0
            elif result == "0-1": value = -1.0
            elif result == "1/2-1/2": value = 0.0
            else: continue
            
            board = game.board()
            for move in game.mainline_moves():
                states.append(board_to_tensor(board))
                values.append(value)
                board.push(move)
            
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Parsed {game_count} games...")
    
    print(f"Total positions: {len(states)}")
    return np.array(states, dtype=np.float32), np.array(values, dtype=np.float32)

def save_dataset(states, values, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/states.npy", states)
    np.save(f"{output_dir}/values.npy", values)
    print(f"Dataset saved to {output_dir}/")

if __name__ == "__main__":
    print("Download a PGN file from lichess.org/games/export/your_username")
    print("Then run: python -m model.dataset path/to/file.pgn")