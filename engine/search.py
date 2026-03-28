import chess
from engine.evaluate import evaluate

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    if maximizing:
        max_score = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score
    else:
        min_score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score

def get_best_move(board, depth=3):
    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth-1, -float('inf'), float('inf'), False)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

if __name__ == "__main__":
    board = chess.Board()
    print("Thinking...")
    move = get_best_move(board, depth=3)
    print("Best move:", move)