import chess
import numpy as np

def board_to_matrix(board):
    # Chuyen 64 o co thanh mang 1 chieu
    matrix = np.zeros(64, dtype=np.int8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            matrix[i] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
    return matrix