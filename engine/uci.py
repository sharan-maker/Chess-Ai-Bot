import chess
import sys
from engine.search import get_best_move

def uci_loop():
    board = chess.Board()
    
    while True:
        msg = input().strip()
        
        if msg == "uci":
            print("id name ChessAiBot")
            print("id author sharan-maker")
            print("uciok")
        
        elif msg == "isready":
            print("readyok")
        
        elif msg == "ucinewgame":
            board.reset()
        
        elif msg.startswith("position"):
            parts = msg.split()
            if "startpos" in parts:
                board.reset()
                if "moves" in parts:
                    moves = parts[parts.index("moves")+1:]
                    for move in moves:
                        board.push_uci(move)
        
        elif msg.startswith("go"):
            move = get_best_move(board, depth=3)
            print(f"bestmove {move}")
        
        elif msg == "quit":
            sys.exit()

if __name__ == "__main__":
    uci_loop()