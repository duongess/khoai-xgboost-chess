from curses import version

import chess
import numpy as np

from config.Setting import PIECE_VALUES, get_play_path, save_game_history

def board_to_matrix(board):
    # Chuyen 64 o co thanh mang 1 chieu
    matrix = np.zeros(64, dtype=np.int8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            matrix[i] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
    return matrix

def game_over(board, color, model_name, version):
    print("Tran dau ket thuc!")
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
        
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"Thach dau AI {model_name}"
    game.headers["White"] = "Human" if color == "white" else model_name
    game.headers["Black"] = model_name if color == "white" else "Human"
    game.headers["Result"] = board.result()
    with open(get_play_path(model_name), "a", encoding="utf-8") as f:
        f.write(str(game) + "\n\n")
    
    save_game_history(model_name, version, color, result)



def extract_features(board: chess.Board):
    features = []
    
    # ==========================================
    # PHẦN 1: 64 Ô CỜ (RAW DATA) - Giữ nguyên như cũ
    # ==========================================
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            features.append(val if piece.color == chess.WHITE else -val)
        else:
            features.append(0)

    # ==========================================
    # PHẦN 2: FEATURE ENGINEERING (Dữ liệu trừu tượng)
    # Lấy góc nhìn của người CẦM QUÂN (Trắng hoặc Đen)
    # ==========================================
    turn = board.turn # True = Trắng, False = Đen
    
    # 1. Chênh lệch vật chất (Vật chất phe ta - phe địch)
    my_material = sum(len(board.pieces(pt, turn)) * val for pt, val in PIECE_VALUES.items())
    opp_material = sum(len(board.pieces(pt, not turn)) * val for pt, val in PIECE_VALUES.items())
    features.append(my_material - opp_material)
    
    # 2. Độ linh hoạt (Mobility - Số nước có thể đi hợp lệ)
    features.append(board.legal_moves.count())
    
    # 3. Kiểm soát trung tâm (Bao nhiêu tia ngắm vào d4, e4, d5, e5)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    my_center_control = sum(1 for sq in center_squares if board.is_attacked_by(turn, sq))
    opp_center_control = sum(1 for sq in center_squares if board.is_attacked_by(not turn, sq))
    features.append(my_center_control - opp_center_control)
    
    # 4. Áp lực lên Vua (Có đang chiếu Vua đối phương không?)
    features.append(1 if board.is_check() else 0)
    
    # 5. An toàn Vua (Có bao nhiêu quân phe ta đứng quanh Vua ta để bảo vệ?)
    my_king_sq = board.king(turn)
    king_safety = 0
    if my_king_sq:
        for sq in board.attacks(my_king_sq):
            p = board.piece_at(sq)
            if p and p.color == turn:
                king_safety += 1
    features.append(king_safety)
    
    # TỔNG CỘNG: 64 cột vị trí + 5 cột Đặc trưng = 69 cột dữ liệu
    return np.array(features, dtype=np.float32)