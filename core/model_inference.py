import chess
import numpy as np
import xgboost as xgb
from core.utils import extract_features
from config.Setting import PIECE_VALUES

# ============================================================
# Cau hinh giai doan van co
# ============================================================
# Neu ban da co ham phat hien phase (opening/middlegame/endgame) trong utils.py
# (theo feature "game phase detection" ban da them), nen dung lai ham do o day
# thay vi nguong fullmove_number don gian nay, de dong bo voi luc train model.
OPENING_MOVE_LIMIT = 10  # tu dieu chinh: so nuoc di (fullmove) duoc coi la khai cuoc


def is_opening_phase(board: chess.Board) -> bool:
    return board.fullmove_number <= OPENING_MOVE_LIMIT


# ============================================================
# Static Exchange Evaluation (SEE) - danh gia chuoi an quan
# ============================================================

def _sorted_attackers(board, square, color, removed):
    """Danh sach (gia_tri, o_co) cua quan `color` dang tan cong `square`, re nhat truoc."""
    out = []
    for sq in board.attackers(color, square):
        if sq in removed:
            continue
        piece = board.piece_at(sq)
        if piece:
            out.append((PIECE_VALUES.get(piece.piece_type, 0), sq))
    out.sort(key=lambda x: x[0])
    return out


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """
    Uoc luong ket qua vat chat (tinh theo PIECE_VALUES) cua chuoi an quan
    bat dau bang `move`. Duong = co loi cho ben di `move`.
    Xap xi chuan SEE (bo qua pin/x-ray phuc tap, du dung cho loc nuoc di).
    """
    if not board.is_capture(move):
        return 0

    to_sq = move.to_square
    victim = board.piece_at(to_sq)
    if victim:
        captured_value = PIECE_VALUES.get(victim.piece_type, 0)
    elif board.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
    else:
        captured_value = 0

    attacker_piece = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

    gains = [captured_value]
    removed = {move.from_square}
    last_attacker_value = attacker_value
    side = not board.turn  # sau khi "di" move, ben tan cong tiep theo la doi phuong

    while True:
        attackers = _sorted_attackers(board, to_sq, side, removed)
        if not attackers:
            break
        val, sq = attackers[0]
        gains.append(last_attacker_value - gains[-1])
        removed.add(sq)
        last_attacker_value = val
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])

    return gains[0]


def see_on_square(board: chess.Board, square: int, attacker_color: bool) -> int:
    """
    SEE tu goc nhin cua attacker_color neu ho chu dong an quan tren `square`.
    Dung de kiem tra xem mot quan co dang "treo" (bi an loi) hay khong.
    """
    target = board.piece_at(square)
    if target is None:
        return 0
    target_value = PIECE_VALUES.get(target.piece_type, 0)

    removed = set()
    attackers = _sorted_attackers(board, square, attacker_color, removed)
    if not attackers:
        return 0

    gains = [target_value]
    val, sq = attackers[0]
    removed.add(sq)
    last_attacker_value = val
    side = not attacker_color

    while True:
        next_attackers = _sorted_attackers(board, square, side, removed)
        if not next_attackers:
            break
        val, sq = next_attackers[0]
        gains.append(last_attacker_value - gains[-1])
        removed.add(sq)
        last_attacker_value = val
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])

    return gains[0]


def find_worst_hang(board: chess.Board, color: bool) -> int:
    """
    Sau khi da di xong 1 nuoc, quet tat ca quan cua `color` xem co quan nao
    dang bi treo (doi phuong an lai loi) khong. Tra ve muc lo nang nhat.
    """
    worst = 0
    for sq, piece in board.piece_map().items():
        if piece.color != color:
            continue
        if board.is_attacked_by(not color, sq):
            loss = see_on_square(board, sq, not color)
            if loss > worst:
                worst = loss
    return worst


def find_tactical_moves(board: chess.Board, legal_moves):
    """Tim nuoc chieu het va cac nuoc an quan loi ro rang (SEE > 0)."""
    mate_moves = []
    winning_captures = {}
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            mate_moves.append(move)
        board.pop()

        if board.is_capture(move):
            see_val = static_exchange_eval(board, move)
            if see_val > 0:
                winning_captures[move] = see_val

    return mate_moves, winning_captures


# ============================================================
# 1-ply batch inference
# ============================================================

def batch_predict_1ply(board: chess.Board, model: xgb.XGBRegressor, legal_moves):
    batch = []
    for move in legal_moves:
        board.push(move)
        features = extract_features(board)
        features = np.append(features, 1)
        batch.append(features)
        board.pop()
    return model.predict(np.array(batch))


# ============================================================
# Smart 1-ply (3 tang) - dung cho trung cuoc / tan cuoc
# ============================================================

# Nguong coi la "treo quan": mac dinh ~1 con Tot. Tang len neu AI van qua nhat
# nguoi trong viec chap nhan doi quan can thiet.
HANG_THRESHOLD = PIECE_VALUES[chess.PAWN]
TACTICAL_BONUS_WEIGHT = 0.01  # trong so nho de tactical chi "day nhe" thu tu, khong de bep XGBoost


def smart_1ply(board: chess.Board, model: xgb.XGBRegressor, legal_moves, ai_scores):
    move_to_score = {m: s for m, s in zip(legal_moves, ai_scores)}
    is_ai_white = board.turn == chess.WHITE
    sign = 1 if is_ai_white else -1

    mate_moves, winning_captures = find_tactical_moves(board, legal_moves)
    if mate_moves:
        move = mate_moves[0]
        print(f"[Tactic] Chieu het -> {board.san(move)}")
        return move

    ranked = sorted(
        legal_moves,
        key=lambda m: (sign * move_to_score[m]) + (TACTICAL_BONUS_WEIGHT * winning_captures.get(m, 0)),
        reverse=True,
    )

    fallback = ranked[0]
    
    # Tinh rui ro hien tai cua minh TRUOC KHI thu di bat ky nuoc nao
    hang_before = find_worst_hang(board, board.turn)

    for move in ranked:
        board.push(move)
        our_color = not board.turn # Mau cua AI hien tai 
        
        # Tinh rui ro cua minh SAU KHI da thuc hien move
        hang_after = find_worst_hang(board, our_color) 
        board.pop()

        # Tinh xem muc do rui ro tang them bao nhieu
        delta_hang = hang_after - hang_before
        
        # Chi coi la blunder neu:
        # 1. Rui ro tang len (delta > 0)
        # 2. Tong rui ro moi vuot qua nguong cho phep (hang_after >= HANG_THRESHOLD)
        is_blunder = (delta_hang > 0) and (hang_after >= HANG_THRESHOLD)

        # Chap nhan nuoc di neu khong phai blunder, hoac nuoc do an loi quan 
        # nhieu hon muc do rui ro bi treo quan
        net_ok = hang_after <= winning_captures.get(move, 0)
        
        if not is_blunder or net_ok:
            print(
                f"[Smart 1-ply] -> {board.san(move)} "
                f"(xgb={move_to_score[move]:.4f}, hang_risk={hang_after})"
            )
            return move

    print(f"[Smart 1-ply] Khong co nuoc an toan tuyet doi, fallback -> {board.san(fallback)}")
    return fallback

# ============================================================
# Entry point
# ============================================================

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    ai_scores = batch_predict_1ply(board, model, legal_moves)
    is_ai_white = board.turn == chess.WHITE

    # --- Khai cuoc: chi nhin 1 nuoc, khong tactical tier, khong blunder filter ---
    if is_opening_phase(board):
        best_idx = int(is_ai_white * np.argmax(ai_scores) + (not is_ai_white) * np.argmin(ai_scores))
        best_move = legal_moves[best_idx]
        print(f"[Opening] 1-ply -> {board.san(best_move)} (score={ai_scores[best_idx]:.4f})")
        return best_move

    # --- Trung cuoc / tan cuoc: 3-tier smart 1-ply ---
    return smart_1ply(board, model, legal_moves, ai_scores)