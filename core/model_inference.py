import chess
import numpy as np
import xgboost as xgb
from core.utils import extract_features
from config.Setting import PIECE_VALUES, MAX_NODES, OPENING_MOVE_LIMIT, ENDGAME_MATERIAL_THRESHOLD, SMART_N_PLY, MIDDLEGAME_BEAM_WIDTH, HANG_PENALTY_WEIGHT, HANG_PENALTY_SCALE


def is_opening_phase(board: chess.Board) -> bool:
    return board.fullmove_number <= OPENING_MOVE_LIMIT

def is_endgame_phase(board: chess.Board) -> bool:
    total_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.piece_type not in (chess.KING, chess.PAWN)
    )
    return total_material <= ENDGAME_MATERIAL_THRESHOLD

# def get_book_move(board):
#     """Bọc Sách khai cuộc (Opening Book) - Ưu tiên số 1"""
#     book_path = DATA_DIR / "Titans.bin" 
#     if not book_path.exists():
#         return None
#     try:
#         with chess.polyglot.open_reader(book_path) as reader:
#             entry = reader.weighted_choice(board)
#             return entry.move
#     except:
#         return None

# ============================================================
# Static Exchange Evaluation (SEE) - danh gia chuoi an quan
# ============================================================

def _sorted_attackers(board, square, color, removed):
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
    if not board.is_capture(move): return 0
    to_sq = move.to_square
    victim = board.piece_at(to_sq)
    
    if victim: captured_value = PIECE_VALUES.get(victim.piece_type, 0)
    elif board.is_en_passant(move): captured_value = PIECE_VALUES[chess.PAWN]
    else: captured_value = 0

    attacker_piece = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

    gains = [captured_value]
    removed = {move.from_square}
    last_attacker_value = attacker_value
    side = not board.turn 

    while True:
        attackers = _sorted_attackers(board, to_sq, side, removed)
        if not attackers: break
        val, sq = attackers[0]
        gains.append(last_attacker_value - gains[-1])
        removed.add(sq)
        last_attacker_value = val
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])

    return gains[0]

def see_on_square(board: chess.Board, square: int, attacker_color: bool) -> int:
    target = board.piece_at(square)
    if target is None: return 0
    target_value = PIECE_VALUES.get(target.piece_type, 0)

    removed = set()
    attackers = _sorted_attackers(board, square, attacker_color, removed)
    if not attackers: return 0

    gains = [target_value]
    val, sq = attackers[0]
    removed.add(sq)
    last_attacker_value = val
    side = not attacker_color

    while True:
        next_attackers = _sorted_attackers(board, square, side, removed)
        if not next_attackers: break
        val, sq = next_attackers[0]
        gains.append(last_attacker_value - gains[-1])
        removed.add(sq)
        last_attacker_value = val
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])

    return gains[0]

def find_worst_hang(board: chess.Board, color: bool) -> int:
    worst = 0
    for sq, piece in board.piece_map().items():
        if piece.color != color: continue
        if board.is_attacked_by(not color, sq):
            loss = see_on_square(board, sq, not color)
            if loss > worst: worst = loss
    return worst

def find_tactical_moves(board: chess.Board, legal_moves):
    mate_moves = []
    winning_captures = {}
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate(): mate_moves.append(move)
        board.pop()

        if board.is_capture(move):
            see_val = static_exchange_eval(board, move)
            if see_val > 0: winning_captures[move] = see_val
    return mate_moves, winning_captures

# ============================================================
# Minimax (chi dung cho tan cuoc)
# ============================================================

eval_cache = {}
node_count = 0

def get_xgb_eval(board: chess.Board, model: xgb.XGBRegressor) -> float:
    global node_count
    node_count += 1
    fen = board.fen()
    if fen in eval_cache: return eval_cache[fen]

    features = extract_features(board)
    features = np.append(features, 1)
    score = model.predict(np.array([features]))[0]
    eval_cache[fen] = score
    return score

def leaf_eval(board: chess.Board, model: xgb.XGBRegressor) -> float:
    score = get_xgb_eval(board, model)
    white_hang = find_worst_hang(board, chess.WHITE)
    black_hang = find_worst_hang(board, chess.BLACK)
    penalty = HANG_PENALTY_WEIGHT * (black_hang - white_hang) / HANG_PENALTY_SCALE
    return score + penalty

def alpha_beta_search(board, model, depth, alpha, beta, is_maximizing, max_budget):
    global node_count
    if depth == 0 or board.is_game_over() or node_count >= max_budget:
        return leaf_eval(board, model)

    ordered_moves = order_moves(board, list(board.legal_moves))

    if is_maximizing:
        max_eval = -float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, False, max_budget)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, True, max_budget)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return min_eval

def score_move(board, move):
    move_score = 0
    if move.promotion is not None:
        move_score += 900 if move.promotion == chess.QUEEN else 300
    if board.is_capture(move):
        victim_piece = board.piece_at(move.to_square)
        victim_val = PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else (PIECE_VALUES[chess.PAWN] if board.is_en_passant(move) else 0)
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0
        move_score += 1000 + (10 * victim_val) - attacker_val
    return move_score

def order_moves(board, legal_moves):
    scored_moves = [(move, score_move(board, move)) for move in legal_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, score in scored_moves]

def minimax_root(board: chess.Board, model: xgb.XGBRegressor, legal_moves) -> chess.Move:
    global node_count
    node_count = 0
    eval_cache.clear()

    mate_moves, _ = find_tactical_moves(board, legal_moves)
    if mate_moves:
        move = mate_moves[0]
        print(f"[Endgame Minimax] Chieu het -> {board.san(move)}")
        return move

    is_ai_white = board.turn == chess.WHITE
    best_move = legal_moves[0]
    current_depth = 1

    while node_count < MAX_NODES:
        temp_best_move = legal_moves[0]
        alpha, beta = -float('inf'), float('inf')
        ordered_moves = order_moves(board, legal_moves)

        if is_ai_white:
            best_score = -float('inf')
            for move in ordered_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, False, MAX_NODES)
                board.pop()
                if score > best_score:
                    best_score = score
                    temp_best_move = move
                alpha = max(alpha, score)
                if node_count >= MAX_NODES: break
        else:
            best_score = float('inf')
            for move in ordered_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, True, MAX_NODES)
                board.pop()
                if score < best_score:
                    best_score = score
                    temp_best_move = move
                beta = min(beta, score)
                if node_count >= MAX_NODES: break

        if node_count < MAX_NODES:
            best_move = temp_best_move
            print(f"[Endgame Minimax] Depth {current_depth} | Nodes {node_count} | -> {board.san(best_move)}")
            current_depth += 1

    print(f"[Endgame Minimax] Chon cuoi -> {board.san(best_move)}")
    return best_move

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
# Smart 1-ply (3 tang) - dung cho trung cuoc 
# ============================================================

HANG_THRESHOLD = PIECE_VALUES[chess.PAWN]
TACTICAL_BONUS_WEIGHT = 0.01

def smart_nply_search(board, model, depth, alpha, beta, is_maximizing, ai_color):
    """Hàm đệ quy N-ply siêu nhẹ dành cho cành nhánh của Trung cuộc"""
    global node_count
    
    # Chạm đáy N-ply hoặc cạn ngân sách: Chốt điểm bằng Node Lá
    if depth == 0 or board.is_game_over() or node_count >= MAX_NODES:
        return leaf_eval(board, model)

    # Khác với Tàn cuộc duyệt toàn bộ, Trung cuộc chỉ xét Top N nước Heuristic
    legal_moves = list(board.legal_moves)
    ordered_moves = order_moves(board, legal_moves)
    beam_moves = ordered_moves[:MIDDLEGAME_BEAM_WIDTH]

    if is_maximizing:
        max_eval = -float('inf')
        for move in beam_moves:
            board.push(move)
            eval_score = smart_nply_search(board, model, depth - 1, alpha, beta, False, ai_color)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha: break 
        return max_eval
    else:
        min_eval = float('inf')
        for move in beam_moves:
            board.push(move)
            eval_score = smart_nply_search(board, model, depth - 1, alpha, beta, True, ai_color)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha: break 
        return min_eval

def get_smart_nply_move(board: chess.Board, model: xgb.XGBRegressor, legal_moves, ai_scores):
    """Hàm gốc (Root) của Smart N-ply: Lọc thông minh rồi mới cho đệ quy"""
    global node_count
    node_count = 0
    move_to_score = {m: s for m, s in zip(legal_moves, ai_scores)}
    is_ai_white = board.turn == chess.WHITE
    ai_color = chess.WHITE if is_ai_white else chess.BLACK
    sign = 1 if is_ai_white else -1

    # Tier 1: Nếu có đòn chiếu bí, đi luôn không cần nghĩ thêm
    mate_moves, winning_captures = find_tactical_moves(board, legal_moves)
    if mate_moves:
        move = mate_moves[0]
        print(f"[Smart {SMART_N_PLY}-ply Tactic] Chiếu hết -> {board.san(move)}")
        return move

    # Xếp hạng tổng thể dựa trên XGBoost tĩnh
    ranked = sorted(
        legal_moves,
        key=lambda m: (sign * move_to_score[m]) + (TACTICAL_BONUS_WEIGHT * winning_captures.get(m, 0)),
        reverse=True,
    )

    fallback = ranked[0]
    safe_candidates = []
    
    # Tính rủi ro treo quân trước khi đi
    hang_before = find_worst_hang(board, board.turn)

    # Tier 2 & 3: Lọc ra các nước an toàn (Không nướng quân)
    for move in ranked:
        board.push(move)
        our_color = not board.turn 
        hang_after = find_worst_hang(board, our_color)
        board.pop()

        delta_hang = hang_after - hang_before
        is_blunder = (delta_hang > 0) and (hang_after >= HANG_THRESHOLD)
        net_ok = hang_after <= winning_captures.get(move, 0)
        
        if not is_blunder or net_ok:
            safe_candidates.append(move)
            # Chỉ lấy đủ số lượng BEAM_WIDTH để đưa vào đệ quy
            if len(safe_candidates) >= MIDDLEGAME_BEAM_WIDTH:
                break

    # Nếu không tìm thấy nước nào an toàn, buộc phải chọn nước ít tệ nhất
    if not safe_candidates:
        print(f"[Smart {SMART_N_PLY}-ply] Tử thủ, fallback -> {board.san(fallback)}")
        return fallback

    # Nếu biến toàn cục SMART_N_PLY = 1, đánh luôn giống hệt bản cũ
    if SMART_N_PLY <= 1:
        chosen_move = safe_candidates[0]
        print(f"[Smart 1-ply] -> {board.san(chosen_move)}")
        return chosen_move

    # --- BẮT ĐẦU ĐÀO SÂU N-PLY CHO CÁC NƯỚC AN TOÀN ---
    print(f"--- [MIDDLEGAME] Ép duyệt sâu Smart {SMART_N_PLY}-Ply cho Top {len(safe_candidates)} nước ---")
    best_move = safe_candidates[0]
    alpha = -float('inf')
    beta = float('inf')
    
    if is_ai_white:
        best_score = -float('inf')
        for move in safe_candidates:
            board.push(move)
            score = smart_nply_search(board, model, SMART_N_PLY - 1, alpha, beta, False, ai_color)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
    else:
        best_score = float('inf')
        for move in safe_candidates:
            board.push(move)
            score = smart_nply_search(board, model, SMART_N_PLY - 1, alpha, beta, True, ai_color)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)

    print(f"[Smart {SMART_N_PLY}-ply] Chốt hạ: {board.san(best_move)} (Đã duyệt {node_count} nodes)")
    return best_move

# ============================================================
# Entry point
# ============================================================

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    ai_scores = batch_predict_1ply(board, model, legal_moves)

    is_ai_white = board.turn == chess.WHITE

    # --- 2. Khai cuộc bằng XGBoost (Nếu out of book) ---
    if is_opening_phase(board):
        if is_ai_white:
            best_idx = int(np.argmax(ai_scores))
        else:
            best_idx = int(np.argmin(ai_scores))
            
        best_move = legal_moves[best_idx]
        print(f"[Opening 1-ply] -> {board.san(best_move)} (score={ai_scores[best_idx]:.4f})")
        return best_move

    # --- 3. Tàn cuộc: Minimax ---
    if is_endgame_phase(board):
        return minimax_root(board, model, legal_moves)

    # --- 4. Trung cuộc: Smart 1-ply ---
    return get_smart_nply_move(board, model, legal_moves, ai_scores)