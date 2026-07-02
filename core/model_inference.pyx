# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import cython
import numpy as np
cimport numpy as cnp
import chess
import xgboost as xgb

from core.utils import extract_features
from config.Setting import PIECE_VALUES, MAX_NODES, OPENING_MOVE_LIMIT, ENDGAME_MATERIAL_THRESHOLD, SMART_N_PLY, MIDDLEGAME_BEAM_WIDTH, TACTICAL_BONUS_WEIGHT

# Dinh nghia cac bien toan cuc kieu C
cdef dict eval_cache = {}
cdef int node_count = 0

cdef bint is_opening_phase(object board):
    return board.fullmove_number <= OPENING_MOVE_LIMIT

cdef bint is_endgame_phase(object board):
    # Kiem tra giai doan tan cuoc. 
    # Canh bao: Neu ban tang PIECE_VALUES x100, phai tang ENDGAME_MATERIAL_THRESHOLD len tuong ung!
    cdef int total_material = 0
    for p in board.piece_map().values():
        if p.piece_type not in (chess.KING, chess.PAWN):
            total_material += PIECE_VALUES.get(p.piece_type, 0)
    return total_material <= ENDGAME_MATERIAL_THRESHOLD

cpdef int material_balance(object board):
    # Dem vat chat cuc nhanh bang bitboard theo thang Centipawn
    cdef int wp = len(board.pieces(chess.PAWN, chess.WHITE))
    cdef int bp = len(board.pieces(chess.PAWN, chess.BLACK))
    cdef int wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    cdef int bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    cdef int wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    cdef int bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    cdef int wr = len(board.pieces(chess.ROOK, chess.WHITE))
    cdef int br = len(board.pieces(chess.ROOK, chess.BLACK))
    cdef int wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    cdef int bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    return ((wp - bp) * 100 + (wn - bn) * 300 + (wb - bb) * 300 + (wr - br) * 500 + (wq - bq) * 900)

cdef list _sorted_attackers(object board, int square, bint color, set removed):
    cdef list out = []
    cdef int sq
    cdef int val
    for sq in board.attackers(color, square):
        if sq in removed:
            continue
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            out.append((val, sq))
    out.sort(key=lambda x: x[0])
    return out

cpdef int static_exchange_eval(object board, object move):
    if not board.is_capture(move): 
        return 0
        
    cdef int to_sq = move.to_square
    victim = board.piece_at(to_sq)
    cdef int captured_value = 0
    
    if victim: 
        captured_value = PIECE_VALUES.get(victim.piece_type, 0)
    elif board.is_en_passant(move): 
        captured_value = PIECE_VALUES[chess.PAWN]

    attacker_piece = board.piece_at(move.from_square)
    cdef int attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

    cdef list gains = [captured_value]
    cdef set removed = {move.from_square}
    cdef int last_attacker_value = attacker_value
    cdef bint side = not board.turn 
    cdef int i
    cdef int last_idx

    while True:
        attackers = _sorted_attackers(board, to_sq, side, removed)
        if not attackers: 
            break
        val, sq = attackers[0]
        
        last_idx = len(gains) - 1
        gains.append(last_attacker_value - gains[last_idx])
        
        removed.add(sq)
        last_attacker_value = val
        side = not side

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])

    return gains[0]

cpdef tuple find_tactical_moves(object board, list legal_moves):
    cdef list mate_moves = []
    cdef dict winning_captures = {}
    cdef int see_val
    
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

cdef float get_xgb_eval(object board, object model):
    global node_count
    node_count += 1
    cdef str fen = board.fen()
    if fen in eval_cache: 
        return eval_cache[fen]

    features = extract_features(board)
    features = np.append(features, 1)
    cdef float score = model.predict(np.array([features]))[0]
    eval_cache[fen] = score
    return score

cdef float leaf_eval(object board, object model):
    # Diem tong hop chuan muc: Vat chat tinh + Diem vi tri XGBoost
    cdef float positional_score = get_xgb_eval(board, model)
    cdef int material_score = material_balance(board)
    return material_score + positional_score

cdef float quiescence_search(object board, object model, float alpha, float beta, bint is_maximizing):
    global node_count
    if node_count >= MAX_NODES:
        return leaf_eval(board, model)

    cdef float stand_pat = leaf_eval(board, model)
    cdef list captures = [m for m in board.legal_moves if board.is_capture(m)]
    cdef list ordered_captures = order_moves(board, captures)
    cdef float score

    if is_maximizing:
        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat

        for move in ordered_captures:
            board.push(move)
            score = quiescence_search(board, model, alpha, beta, False)
            board.pop()
            if score > alpha: alpha = score
            if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        if stand_pat < beta: beta = stand_pat

        for move in ordered_captures:
            board.push(move)
            score = quiescence_search(board, model, alpha, beta, True)
            board.pop()
            if score < beta: beta = score
            if alpha >= beta: break
        return beta

cdef float alpha_beta_search(object board, object model, int depth, float alpha, float beta, bint is_maximizing, int max_budget):
    global node_count
    if board.is_game_over() or node_count >= max_budget:
        return leaf_eval(board, model)

    # ĐIỂM CHỐT: Tàn cuộc cũng gọi QS thay vì dừng ở node lá tĩnh
    if depth == 0:
        return quiescence_search(board, model, alpha, beta, is_maximizing)

    cdef list ordered_moves = order_moves(board, list(board.legal_moves))
    cdef float max_eval, min_eval, eval_score

    if is_maximizing:
        max_eval = -float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, False, max_budget)
            board.pop()
            if eval_score > max_eval: max_eval = eval_score
            if eval_score > alpha: alpha = eval_score
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, True, max_budget)
            board.pop()
            if eval_score < min_eval: min_eval = eval_score
            if eval_score < beta: beta = eval_score
            if beta <= alpha: break
        return min_eval

cdef int score_move(object board, object move):
    cdef int move_score = 0
    if move.promotion is not None:
        move_score += 900 if move.promotion == chess.QUEEN else 300
    if board.is_capture(move):
        victim_piece = board.piece_at(move.to_square)
        victim_val = PIECE_VALUES.get(victim_piece.piece_type, 0) if victim_piece else (PIECE_VALUES[chess.PAWN] if board.is_en_passant(move) else 0)
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0
        move_score += 1000 + (10 * victim_val) - attacker_val
    return move_score

cdef list order_moves(object board, list legal_moves):
    cdef list scored_moves = [(move, score_move(board, move)) for move in legal_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, score in scored_moves]

cpdef object minimax_root(object board, object model, list legal_moves):
    global node_count
    node_count = 0
    eval_cache.clear()

    cdef list mate_moves
    mate_moves, _ = find_tactical_moves(board, legal_moves)
    if mate_moves:
        move = mate_moves[0]
        print(f"[Endgame Minimax] Chieu het -> {board.san(move)}")
        return move

    cdef bint is_ai_white = board.turn == chess.WHITE
    best_move = legal_moves[0]
    cdef int current_depth = 1
    cdef float alpha, beta, best_score, score

    while node_count < MAX_NODES:
        temp_best_move = legal_moves[0]
        alpha = -float('inf')
        beta = float('inf')
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
                if score > alpha: alpha = score
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
                if score < beta: beta = score
                if node_count >= MAX_NODES: break

        if node_count < MAX_NODES:
            best_move = temp_best_move
            print(f"[Endgame Minimax] Depth {current_depth} | Nodes {node_count} | -> {board.san(best_move)}")
            current_depth += 1

    print(f"[Endgame Minimax] Chon cuoi -> {board.san(best_move)}")
    return best_move

cpdef object batch_predict_1ply(object board, object model, list legal_moves):
    # Tinh luon ca vat chat o 1-ply de rank chuan xac ngay tu dau
    cdef list batch = []
    cdef list materials = []
    for move in legal_moves:
        board.push(move)
        features = extract_features(board)
        features = np.append(features, 1)
        batch.append(features)
        materials.append(material_balance(board))
        board.pop()
    
    pos_scores = model.predict(np.array(batch))
    return pos_scores + np.array(materials)

cdef float smart_nply_search(object board, object model, int depth, float alpha, float beta, bint is_maximizing, bint ai_color):
    global node_count
    
    if board.is_game_over() or node_count >= MAX_NODES:
        return leaf_eval(board, model)

    # ĐIỂM CHỐT: Trung cuộc chạm đáy cũng gọi QS
    if depth == 0:
        return quiescence_search(board, model, alpha, beta, is_maximizing)

    cdef list legal_moves = list(board.legal_moves)
    cdef list ordered_moves = order_moves(board, legal_moves)
    cdef list beam_moves = ordered_moves[:MIDDLEGAME_BEAM_WIDTH]
    cdef float max_eval, min_eval, eval_score

    if is_maximizing:
        max_eval = -float('inf')
        for move in beam_moves:
            board.push(move)
            eval_score = smart_nply_search(board, model, depth - 1, alpha, beta, False, ai_color)
            board.pop()
            if eval_score > max_eval: max_eval = eval_score
            if eval_score > alpha: alpha = eval_score
            if beta <= alpha: break 
        return max_eval
    else:
        min_eval = float('inf')
        for move in beam_moves:
            board.push(move)
            eval_score = smart_nply_search(board, model, depth - 1, alpha, beta, True, ai_color)
            board.pop()
            if eval_score < min_eval: min_eval = eval_score
            if eval_score < beta: beta = eval_score
            if beta <= alpha: break 
        return min_eval

def get_smart_nply_move(board, model, list legal_moves, object ai_scores):
    global node_count
    node_count = 0
    
    cdef dict move_to_score = {m: s for m, s in zip(legal_moves, ai_scores)}
    cdef bint is_ai_white = board.turn == chess.WHITE
    cdef bint ai_color = chess.WHITE if is_ai_white else chess.BLACK
    cdef int sign = 1 if is_ai_white else -1

    cdef list mate_moves
    cdef dict winning_captures
    mate_moves, winning_captures = find_tactical_moves(board, legal_moves)
    
    if mate_moves:
        move = mate_moves[0]
        print(f"[Smart {SMART_N_PLY}-ply Tactic] Chiếu hết -> {board.san(move)}")
        return move

    cdef list ranked = sorted(
        legal_moves,
        key=lambda m: (sign * move_to_score[m]) + (TACTICAL_BONUS_WEIGHT * winning_captures.get(m, 0)),
        reverse=True,
    )

    cdef list beam_candidates = ranked[:MIDDLEGAME_BEAM_WIDTH]

    if SMART_N_PLY <= 1:
        chosen_move = beam_candidates[0]
        print(f"[Smart 1-ply] -> {board.san(chosen_move)}")
        return chosen_move

    print(f"--- [MIDDLEGAME] Ép duyệt sâu Smart {SMART_N_PLY}-Ply cho Top {len(beam_candidates)} nước ---")
    
    best_move = beam_candidates[0]
    cdef float alpha = -float('inf')
    cdef float beta = float('inf')
    cdef float best_score, score
    
    if is_ai_white:
        best_score = -float('inf')
        for move in beam_candidates:
            board.push(move)
            score = smart_nply_search(board, model, SMART_N_PLY - 1, alpha, beta, False, ai_color)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha: alpha = score
    else:
        best_score = float('inf')
        for move in beam_candidates:
            board.push(move)
            score = smart_nply_search(board, model, SMART_N_PLY - 1, alpha, beta, True, ai_color)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            if score < beta: beta = score

    print(f"[Smart {SMART_N_PLY}-ply] Chốt hạ: {board.san(best_move)} (Đã duyệt {node_count} nodes)")
    return best_move

def get_ai_move(object board, object model):
    cdef list legal_moves = list(board.legal_moves)
    if not legal_moves: 
        return None
        
    ai_scores = batch_predict_1ply(board, model, legal_moves)
    cdef bint is_ai_white = board.turn == chess.WHITE
    cdef int best_idx

    if is_opening_phase(board):
        if is_ai_white:
            best_idx = int(np.argmax(ai_scores))
        else:
            best_idx = int(np.argmin(ai_scores))
            
        best_move = legal_moves[best_idx]
        print(f"[Opening 1-ply] -> {board.san(best_move)} (score={ai_scores[best_idx]:.4f})")
        return best_move

    if is_endgame_phase(board):
        return minimax_root(board, model, legal_moves)

    return get_smart_nply_move(board, model, legal_moves, ai_scores)