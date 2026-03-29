import chess
import numpy as np

def board_to_tensor(board):
    tensor = np.zeros((119, 8, 8), dtype=np.float32)
    
    piece_idx = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            idx = piece_idx[(piece.piece_type, piece.color)]
            tensor[idx][row][col] = 1.0
    
    tensor[112] = float(board.turn)
    tensor[113] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[114] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[115] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[116] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    return tensor

if __name__ == "__main__":
    board = chess.Board()
    tensor = board_to_tensor(board)
    print("Tensor shape:", tensor.shape)
    print("Non-zero planes:", np.count_nonzero(tensor.sum(axis=(1,2))))

'''python -m engine.encoder'''