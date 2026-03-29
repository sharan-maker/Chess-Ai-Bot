import chess
import chess.engine
import sys
sys.path.insert(0, '.')
from engine.search import get_best_move

STOCKFISH_PATH = r"C:\Users\sharan kumar H\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

def benchmark(num_games=10):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": 0})
    
    wins, losses, draws = 0, 0, 0

    for game in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = get_best_move(board, depth=3)
            else:
                result = engine.play(board, chess.engine.Limit(depth=1))
                move = result.move
            board.push(move)
        
        outcome = board.result()
        if outcome == "1-0": wins += 1
        elif outcome == "0-1": losses += 1
        else: draws += 1
        print(f"Game {game+1}: {outcome}")

    engine.quit()
    print(f"\nResults: {wins}W / {losses}L / {draws}D")

if __name__ == "__main__":
    benchmark()