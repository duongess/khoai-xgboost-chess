from curses import version

import chess
import numpy as np

from config.Setting import (
    PAWN_PST, KNIGHT_PST, BISHOP_PST, PIECE_VALUES,
    ROOK_PST, QUEEN_PST, KING_PST,
    get_play_path, save_game_history
)

KING_ENDGAME_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]


def board_to_matrix(board):
    matrix = np.zeros(64, dtype=np.int8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            matrix[i] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
    return matrix


def game_over(board, color, model_name, version):
    print("Tran dau ket thuc!")
    result = board.result()
    if result == "1-0" and board.turn == chess.BLACK:
        print("Chuc mung! Ban da thang.")
    elif result == "0-1" and board.turn == chess.WHITE:
        print("Chuc mung! Ban da thang.")
    elif result == "1/2-1/2":
        print("Tran dau hoa.")
    else:
        print("Ban da thua. Hay co gang lan sau!")

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"Thach dau AI {model_name}"
    game.headers["White"] = "Human" if color == "white" else model_name
    game.headers["Black"] = model_name if color == "white" else "Human"
    game.headers["Result"] = board.result()
    with open(get_play_path(model_name), "a", encoding="utf-8") as f:
        f.write(str(game) + "\n\n")

    save_game_history(model_name, version, color, result)


def get_game_phase(board) -> float:
    """
    Tinh giai doan van co dua tren vat chat con lai (tru tot & vua).
    1.0 = khai cuoc day du, 0.0 = tan cuoc hoan toan.
    """
    max_material = 2 * (2*3 + 2*3 + 2*5 + 9)  # = 46
    current = 0
    for pt, val in [(chess.KNIGHT, 3), (chess.BISHOP, 3),
                    (chess.ROOK, 5), (chess.QUEEN, 9)]:
        current += len(board.pieces(pt, chess.WHITE)) * val
        current += len(board.pieces(pt, chess.BLACK)) * val
    return min(current / max_material, 1.0)


def calculate_pst_score(board, turn, phase: float = 1.0) -> float:
    """
    Tinh diem PST cho phe chi dinh.
    Vua dung Tapered Eval: blend PST trung cuoc vs tan cuoc theo phase.
    """
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == turn:
            pst_sq = square if turn == chess.WHITE else chess.square_mirror(square)
            if piece.piece_type == chess.PAWN:
                score += PAWN_PST[pst_sq]
            elif piece.piece_type == chess.KNIGHT:
                score += KNIGHT_PST[pst_sq]
            elif piece.piece_type == chess.BISHOP:
                score += BISHOP_PST[pst_sq]
            elif piece.piece_type == chess.ROOK:
                score += ROOK_PST[pst_sq]
            elif piece.piece_type == chess.QUEEN:
                score += QUEEN_PST[pst_sq]
            elif piece.piece_type == chess.KING:
                mid_val = KING_PST[pst_sq]
                end_val = KING_ENDGAME_PST[pst_sq]
                score += int(phase * mid_val + (1 - phase) * end_val)
    return score / 100.0


def evaluate_rooks(board, turn) -> float:
    score = 0
    for square in board.pieces(chess.ROOK, turn):
        file_mask = chess.BB_FILES[chess.square_file(square)]
        my_pawns  = board.pieces(chess.PAWN, turn) & file_mask
        opp_pawns = board.pieces(chess.PAWN, not turn) & file_mask
        if not my_pawns and not opp_pawns:
            score += 1.5
        elif not my_pawns:
            score += 0.5
    return score


def evaluate_king_attack(board, turn) -> float:
    score = 0
    opp_king_sq = board.king(not turn)
    if opp_king_sq is None:
        return 0
    king_zone = board.attacks(opp_king_sq) | {opp_king_sq}
    for sq in king_zone:
        if board.is_attacked_by(turn, sq):
            score += 1
    return score


# ═══════════════════════════════════════════
# NHOM FEATURE MOI: DANH GIA CUC BO TUNG QUAN
# ═══════════════════════════════════════════

def evaluate_piece_safety(board, turn):
    """
    Phan loai tung quan cua 'turn':
      hung (treo)  : bi tan cong ma KHONG duoc bao ve
      attacked     : bi tan cong nhung CO quan bao ve
    Tra ve (hung_val, attacked_val).
    """
    hung_val = 0
    attacked_val = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != turn:
            continue
        val = PIECE_VALUES.get(p.piece_type, 0)
        is_attacked = board.is_attacked_by(not turn, sq)
        is_defended = board.is_attacked_by(turn, sq)
        if is_attacked and not is_defended:
            hung_val += val
        elif is_attacked:
            attacked_val += val
    return hung_val, attacked_val


def evaluate_most_valuable_hanging(board, turn) -> float:
    """
    Gia tri cua quan CO GIA TRI CAO NHAT dang bi treo.
    Feature quan trong nhat de AI khong de mat hau/xe vo ly.
    """
    max_hung = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != turn:
            continue
        if board.is_attacked_by(not turn, sq) and not board.is_attacked_by(turn, sq):
            val = PIECE_VALUES.get(p.piece_type, 0)
            if val > max_hung:
                max_hung = val
    return float(max_hung)


def evaluate_pawn_structure(board, turn):
    """
    Phan tich cau truc tot:
      doubled  : tot doi (2 tot cung cot) -> diem tru
      isolated : tot co lap (khong tot hang xom) -> diem tru
      passed   : tot thong (khong tot dich chan phia truoc) -> diem cong
    Tra ve (doubled, isolated, passed).
    """
    doubled  = 0
    isolated = 0
    passed   = 0

    my_pawns  = board.pieces(chess.PAWN, turn)
    opp_pawns = board.pieces(chess.PAWN, not turn)

    files_count = {}
    for sq in my_pawns:
        f = chess.square_file(sq)
        files_count[f] = files_count.get(f, 0) + 1

    for sq in my_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        if files_count[f] > 1:
            doubled += 1

        left  = any(chess.square_file(s) == f - 1 for s in my_pawns) if f > 0 else False
        right = any(chess.square_file(s) == f + 1 for s in my_pawns) if f < 7 else False
        if not left and not right:
            isolated += 1

        is_passed = True
        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if abs(opp_f - f) <= 1:
                if turn == chess.WHITE and opp_r > r:
                    is_passed = False
                    break
                if turn == chess.BLACK and opp_r < r:
                    is_passed = False
                    break
        if is_passed:
            passed += 1

    return doubled, isolated, passed


def evaluate_knight_outpost(board, turn) -> float:
    """
    Ma dung o outpost: o khong bi tot dich tan cong,
    co tot minh bao ve, nam o nua ban dich.
    """
    score = 0
    opp_pawns = board.pieces(chess.PAWN, not turn)
    for sq in board.pieces(chess.KNIGHT, turn):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        in_enemy_half = (r >= 4) if turn == chess.WHITE else (r <= 3)
        if not in_enemy_half:
            continue
        attacked_by_pawn = False
        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if turn == chess.WHITE:
                if abs(opp_f - f) == 1 and opp_r == r + 1:
                    attacked_by_pawn = True
                    break
            else:
                if abs(opp_f - f) == 1 and opp_r == r - 1:
                    attacked_by_pawn = True
                    break
        if not attacked_by_pawn and board.is_attacked_by(turn, sq):
            score += 1
    return float(score)


def evaluate_bishop_pair(board, turn) -> float:
    """Cap tuong: 2 tuong con nguyen = loi the chien luoc ro ret."""
    return 1.0 if len(board.pieces(chess.BISHOP, turn)) >= 2 else 0.0


def evaluate_king_safety_zone(board, turn) -> float:
    """
    Dem so o trong vung 3x3 quanh vua minh dang bi dich kiem soat.
    Phat them neu vua chua nhap thanh (con o cot e).
    """
    my_king_sq = board.king(turn)
    if my_king_sq is None:
        return 0.0
    danger = 0
    king_zone = board.attacks(my_king_sq) | {my_king_sq}
    for sq in king_zone:
        if board.is_attacked_by(not turn, sq):
            danger += 1
    if chess.square_file(my_king_sq) == 4:  # Cot e -> chua nhap thanh
        danger += 2
    return float(danger)


def evaluate_space_control(board, turn) -> float:
    """So o o nua ban dich ma phe 'turn' dang kiem soat."""
    target_ranks = range(4, 8) if turn == chess.WHITE else range(0, 4)
    return float(sum(
        1 for sq in chess.SQUARES
        if chess.square_rank(sq) in target_ranks and board.is_attacked_by(turn, sq)
    ))


# ═══════════════════════════════════════════════════════════════
#  extract_features: 93 features
#
#  [0..63]  64 o co
#  [64]     material diff
#  [65]     mobility
#  [66]     center control diff
#  [67]     is_check
#  [68]     king safety cu (backward compat)
#  [69]     pst diff (tapered)
#  [70]     rook open file diff
#  [71]     king attack diff
#  [72]     game phase
#  [73]     my hung total val
#  [74]     opp hung total val
#  [75]     my most valuable hung
#  [76]     opp most valuable hung
#  [77]     my attacked-but-defended val
#  [78]     opp attacked-but-defended val
#  [79]     my doubled pawns
#  [80]     opp doubled pawns
#  [81]     my isolated pawns
#  [82]     opp isolated pawns
#  [83]     my passed pawns
#  [84]     opp passed pawns
#  [85]     my knight outposts
#  [86]     opp knight outposts
#  [87]     my bishop pair
#  [88]     opp bishop pair
#  [89]     my space control
#  [90]     opp space control
#  [91]     my king danger zone
#  [92]     opp king danger zone
# ═══════════════════════════════════════════════════════════════

def extract_features(board: chess.Board) -> np.ndarray:
    features = []

    # 1. 64 o co
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            features.append(val if piece.color == chess.WHITE else -val)
        else:
            features.append(0)

    turn  = board.turn
    phase = get_game_phase(board)

    # 2. Dac trung tong quat
    my_mat  = sum(len(board.pieces(pt, turn))     * v for pt, v in PIECE_VALUES.items())
    opp_mat = sum(len(board.pieces(pt, not turn)) * v for pt, v in PIECE_VALUES.items())
    features.append(my_mat - opp_mat)                     # [64]
    features.append(board.legal_moves.count())            # [65]

    cs = [chess.D4, chess.E4, chess.D5, chess.E5]
    features.append(
        sum(1 for sq in cs if board.is_attacked_by(turn,     sq)) -
        sum(1 for sq in cs if board.is_attacked_by(not turn, sq))
    )                                                      # [66]
    features.append(1 if board.is_check() else 0)         # [67]

    my_king_sq = board.king(turn)
    ks_old = 0
    if my_king_sq:
        for sq in board.attacks(my_king_sq):
            p = board.piece_at(sq)
            if p and p.color == turn:
                ks_old += 1
    features.append(ks_old)                               # [68]

    # 3. PST tapered + Xe + Tan cong vua
    features.append(
        calculate_pst_score(board, turn, phase) -
        calculate_pst_score(board, not turn, phase)
    )                                                      # [69]
    features.append(
        evaluate_rooks(board, turn) - evaluate_rooks(board, not turn)
    )                                                      # [70]
    features.append(
        evaluate_king_attack(board, turn) - evaluate_king_attack(board, not turn)
    )                                                      # [71]
    features.append(phase)                                 # [72]

    # 4. An toan quan cuc bo
    my_hung,  my_atk  = evaluate_piece_safety(board, turn)
    opp_hung, opp_atk = evaluate_piece_safety(board, not turn)
    features.append(my_hung)                              # [73]
    features.append(opp_hung)                             # [74]
    features.append(evaluate_most_valuable_hanging(board, turn))      # [75]
    features.append(evaluate_most_valuable_hanging(board, not turn))  # [76]
    features.append(my_atk)                               # [77]
    features.append(opp_atk)                              # [78]

    # 5. Cau truc tot
    my_d,  my_i,  my_p  = evaluate_pawn_structure(board, turn)
    opp_d, opp_i, opp_p = evaluate_pawn_structure(board, not turn)
    features.extend([my_d, opp_d, my_i, opp_i, my_p, opp_p])  # [79..84]

    # 6. Quan manh & khong gian
    features.append(evaluate_knight_outpost(board, turn))           # [85]
    features.append(evaluate_knight_outpost(board, not turn))       # [86]
    features.append(evaluate_bishop_pair(board, turn))              # [87]
    features.append(evaluate_bishop_pair(board, not turn))          # [88]
    features.append(evaluate_space_control(board, turn))            # [89]
    features.append(evaluate_space_control(board, not turn))        # [90]

    # 7. An toan vua chi tiet
    features.append(evaluate_king_safety_zone(board, turn))         # [91]
    features.append(evaluate_king_safety_zone(board, not turn))     # [92]

    return np.array(features, dtype=np.float32)