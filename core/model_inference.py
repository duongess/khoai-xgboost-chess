import chess
import numpy as np
import xgboost as xgb
from config.Setting import MAX_NODES, PIECE_VALUES
from core.utils import extract_features

eval_cache = {}
node_count = 0

def get_xgb_eval(board, model):
    global node_count
    node_count += 1
    
    fen = board.fen()
    if fen in eval_cache:
        return eval_cache[fen]
        
    features = extract_features(board)
    features = np.append(features, 1) 
    
    score = model.predict(np.array([features]))[0]
    
    eval_cache[fen] = score
    return score

def score_move(board, move):
    """
    Cham diem nhanh nuoc di de sap xep. Diem cang cao, cang duoc duyet truoc.
    """
    move_score = 0
    
    # 1. Nuoc phong cap (Uu tien rat cao)
    if move.promotion is not None:
        if move.promotion == chess.QUEEN:
            move_score += 900
        else:
            move_score += 300
            
    # 2. Nuoc an quan (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
    if board.is_capture(move):
        # Tim quan bi an (Victim)
        victim_piece = board.piece_at(move.to_square)
        if victim_piece:
            victim_val = PIECE_VALUES.get(victim_piece.piece_type, 0)
        elif board.is_en_passant(move):
            victim_val = PIECE_VALUES[chess.PAWN]
        else:
            victim_val = 0
            
        # Tim quan an (Attacker)
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0
        
        # Cong thuc MVV-LVA co ban: Nhan 10 cho quan bi an de uu tien muc tieu to, tru di quan an
        # VD: Tot (10) an Hau (90) = 900 - 10 = 890 (Uu tien cao nhat)
        # VD: Hau (90) an Tot (10) = 100 - 90 = 10 (Uu tien thap nhat trong dam an quan)
        move_score += 1000 + (10 * victim_val) - attacker_val
        
    return move_score

def order_moves(board, legal_moves):
    """
    Sap xep danh sach nuoc di hop le dua tren diem score_move.
    """
    # Tao danh sach tuple (nuoc_di, diem)
    scored_moves = [(move, score_move(board, move)) for move in legal_moves]
    
    # Sap xep giam dan theo diem
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    # Tra ve danh sach cac nuoc di da sap xep
    return [move for move, score in scored_moves]

# Cach tich hop vao ham alpha_beta_search cua ban:
def alpha_beta_search(board, model, depth, alpha, beta, is_maximizing, current_max_nodes):
    global node_count
    
    if depth == 0 or board.is_game_over() or node_count >= current_max_nodes:
        return get_xgb_eval(board, model)

    legal_moves = list(board.legal_moves)
    
    # THAY THE TODO BANG DONG NAY
    ordered_moves = order_moves(board, legal_moves)

    if is_maximizing:
        max_eval = -float('inf')
        # Duyet qua ordered_moves thay vi legal_moves
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, False, current_max_nodes)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break 
        return max_eval
        
    else:
        min_eval = float('inf')
        # Duyet qua ordered_moves thay vi legal_moves
        for move in ordered_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, True, current_max_nodes)
            board.pop()
            
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break 
        return min_eval

def calculate_dynamic_budget(board, max_nodes_limit):
    """
    Tinh toan ngan sach node dong.
    Khoi diem la MAX_NODES // 10.
    Tang dan sau moi ply (nua nuoc di). Tieu chuan tang la 5% moi ply.
    Se dat dinh MAX_NODES o khoang nuoc di thu 10-15 cua van co.
    """
    base_budget = max_nodes_limit // 10
    current_ply = board.ply()
    
    # Tang dan ngan sach, toi da khong vuot qua max_nodes_limit
    added_budget = int(max_nodes_limit * 0.05 * current_ply)
    dynamic_budget = min(max_nodes_limit, base_budget + added_budget)
    
    return dynamic_budget

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    global node_count
    node_count = 0 
    eval_cache.clear() 
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    best_move = None
    is_ai_white = board.turn == chess.WHITE
    
    # Tinh toan ngan sach cho nuoc di hien tai
    current_max_nodes = calculate_dynamic_budget(board, MAX_NODES)
    
    current_depth = 1
    print(f"Bat dau suy nghi. Ngan sach dong: {current_max_nodes}/{MAX_NODES} nodes.")
    
    while node_count < current_max_nodes:
        temp_best_move = legal_moves[0] 
        alpha = -float('inf')
        beta = float('inf')
        
        if is_ai_white:
            best_score = -float('inf')
            for move in legal_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, False, current_max_nodes)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    temp_best_move = move
                alpha = max(alpha, score)
                
                if node_count >= current_max_nodes: break
        else:
            best_score = float('inf')
            for move in legal_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, True, current_max_nodes)
                board.pop()
                
                if score < best_score:
                    best_score = score
                    temp_best_move = move
                beta = min(beta, score)
                
                if node_count >= current_max_nodes: break
                
        if node_count < current_max_nodes:
            best_move = temp_best_move
            print(f"Hoan thanh Depth {current_depth} | Da dung {node_count}/{current_max_nodes} nodes | Tam chon: {board.san(best_move)}")
            current_depth += 1
        else:
            print(f"Het ngan sach khi dang duyet Depth {current_depth}.")
            if best_move is None:
                best_move = temp_best_move
            
    print(f"=> QUYET DINH: {board.san(best_move)} (Da duyet tong cong {node_count} trang thai)")
    return best_move