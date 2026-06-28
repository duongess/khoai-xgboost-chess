from curses import version

import chess
import numpy as np

from config.Setting import PAWN_PST, KNIGHT_PST, BISHOP_PST,  PIECE_VALUES, ROOK_PST, QUEEN_PST, KING_PST, get_play_path, save_game_history

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



def calculate_pst_score(board, turn):
    # Tính điểm vị trí quân cờ (PST) cho phe chỉ định
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == turn:
            pst_square = square if turn == chess.WHITE else chess.square_mirror(square)
            
            if piece.piece_type == chess.PAWN: score += PAWN_PST[pst_square]
            elif piece.piece_type == chess.KNIGHT: score += KNIGHT_PST[pst_square]
            elif piece.piece_type == chess.BISHOP: score += BISHOP_PST[pst_square]
            elif piece.piece_type == chess.ROOK: score += ROOK_PST[pst_square]
            elif piece.piece_type == chess.QUEEN: score += QUEEN_PST[pst_square]
            elif piece.piece_type == chess.KING: score += KING_PST[pst_square]
            
    return score / 100.0

def evaluate_rooks(board, turn):
    # Đánh giá sức mạnh Xe trên cột mở hoặc nửa mở bằng Bitboard
    score = 0
    for square in board.pieces(chess.ROOK, turn):
        file_idx = chess.square_file(square)
        file_mask = chess.BB_FILES[file_idx]
        
        my_pawns = board.pieces(chess.PAWN, turn) & file_mask
        opp_pawns = board.pieces(chess.PAWN, not turn) & file_mask
        
        if not my_pawns and not opp_pawns:
            score += 1.5 
        elif not my_pawns:
            score += 0.5 
    return score

def evaluate_king_attack(board, turn):
    # Đếm số tia ngắm tấn công vào khu vực xung quanh Vua đối phương
    score = 0
    opp_king_sq = board.king(not turn)
    if opp_king_sq is None: 
        return 0
    
    king_zone = board.attacks(opp_king_sq) | {opp_king_sq}
    for sq in king_zone:
        if board.is_attacked_by(turn, sq):
            score += 1
    return score

def extract_features(board: chess.Board):
    features = []
    
    # 1. Quet 64 o co tinh
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            features.append(val if piece.color == chess.WHITE else -val)
        else:
            features.append(0)

    # Luu y: turn hien tai la cua phe CHUAN BI DI
    # (Neu AI vua push(move) xong, turn la cua doi thu)
    turn = board.turn
    
    # 2. Dac trung tinh co ban
    my_material = sum(len(board.pieces(pt, turn)) * val for pt, val in PIECE_VALUES.items())
    opp_material = sum(len(board.pieces(pt, not turn)) * val for pt, val in PIECE_VALUES.items())
    features.append(my_material - opp_material)
    
    features.append(board.legal_moves.count())
    
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    my_center_control = sum(1 for sq in center_squares if board.is_attacked_by(turn, sq))
    opp_center_control = sum(1 for sq in center_squares if board.is_attacked_by(not turn, sq))
    features.append(my_center_control - opp_center_control)
    
    features.append(1 if board.is_check() else 0)
    
    my_king_sq = board.king(turn)
    king_safety = 0
    if my_king_sq:
        for sq in board.attacks(my_king_sq):
            p = board.piece_at(sq)
            if p and p.color == turn:
                king_safety += 1
    features.append(king_safety)
    
    # 3. Dac trung dong (PST, Xe, Tan cong Vua)
    my_pst = calculate_pst_score(board, turn)
    opp_pst = calculate_pst_score(board, not turn)
    features.append(my_pst - opp_pst)
    
    my_rook_score = evaluate_rooks(board, turn)
    opp_rook_score = evaluate_rooks(board, not turn)
    features.append(my_rook_score - opp_rook_score)
    
    my_king_attack = evaluate_king_attack(board, turn)
    opp_king_attack = evaluate_king_attack(board, not turn)
    features.append(my_king_attack - opp_king_attack)
    
    # 4. Dac trung phong ngu (Quan dang bi de doa)
    ai_color = not turn
    opp_color = turn
    
    ai_hanging_val = 0
    opp_hanging_val = 0
    
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            if p.color == ai_color and board.is_attacked_by(opp_color, sq):
                ai_hanging_val += PIECE_VALUES.get(p.piece_type, 0)
            elif p.color == opp_color and board.is_attacked_by(ai_color, sq):
                opp_hanging_val += PIECE_VALUES.get(p.piece_type, 0)
                
    features.append(ai_hanging_val - opp_hanging_val)
    
    return np.array(features, dtype=np.float32)