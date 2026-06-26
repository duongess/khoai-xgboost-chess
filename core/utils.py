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

def check_game_over(board):
    if board.is_game_over():
        result = board.result()
        # 1-0: Trắng thắng, 0-1: Đen thắng, 1/2-1/2: Hòa
        if result == "1-0" and board.turn == chess.BLACK: # Người chơi cầm trắng
            print("Chúc mừng! Bạn đã thắng.")
        elif result == "0-1" and board.turn == chess.WHITE: # Người chơi cầm đen
            print("Chúc mừng! Bạn đã thắng.")
        elif result == "1/2-1/2":
            print("Trận đấu hòa.")
        else:
            print("Bạn đã thua. Hãy cố gắng lần sau!")
        return True
    return False