import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999
        return 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return score

if __name__ == "__main__":
    board = chess.Board()
    print("Starting position score:", evaluate(board))