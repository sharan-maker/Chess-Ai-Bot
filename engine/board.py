import chess

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        self.board.reset()
        return self.get_state()
    
    def get_state(self):
        return self.board.fen()
    
    def get_legal_moves(self):
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        self.board.push(move)
    
    def undo_move(self):
        self.board.pop()
    
    def is_game_over(self):
        return self.board.is_game_over()
    
    def get_result(self):
        result = self.board.result()
        if result == "1-0": return 1
        if result == "0-1": return -1
        return 0
    
    def print_board(self):
        print(self.board)

if __name__ == "__main__":
    env = ChessEnv()
    env.reset()
    env.print_board()
    print("Legal moves:", len(env.get_legal_moves()))