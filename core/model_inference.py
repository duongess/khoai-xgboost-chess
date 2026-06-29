import chess
import numpy as np
import xgboost as xgb
from config.Setting import MAX_NODES
from core.utils import extract_features

# 1. Bộ nhớ đệm (Transposition Table) - Cực kỳ quan trọng để không gọi XGBoost lại cho thế cờ đã gặp
eval_cache = {}

# 2. Biến toàn cục để đếm tổng số thế cờ đã duyệt
node_count = 0

def get_xgb_eval(board, model):
    """Hàm bọc XGBoost có sử dụng cache để tăng tốc Inference đơn lẻ"""
    global node_count
    node_count += 1
    
    fen = board.fen()
    if fen in eval_cache:
        return eval_cache[fen]
        
    features = extract_features(board)
    features = np.append(features, 1) # is_style_focus = 1
    
    # Predict từng node một (Single Inference)
    score = model.predict(np.array([features]))[0]
    
    # Cần điều chỉnh nếu mô hình của bạn chấm điểm theo phe Trắng/Đen
    # Giả sử mô hình trả về điểm dương cho Trắng có lợi, âm cho Đen có lợi
    eval_cache[fen] = score
    return score

def alpha_beta_search(board, model, depth, alpha, beta, is_maximizing):
    """Thuật toán Minimax + Cắt tỉa Alpha-Beta"""
    global node_count
    
    # Dừng nếu cạn độ sâu, hết game, hoặc HẾT NGÂN SÁCH NODE
    if depth == 0 or board.is_game_over() or node_count >= MAX_NODES:
        return get_xgb_eval(board, model)

    legal_moves = list(board.legal_moves)
    
    # MẸO: Sắp xếp nước đi (Move Ordering) ở đây sẽ giúp cắt tỉa nhanh gấp 3 lần!
    # Ví dụ: Đưa các nước is_capture() lên đầu mảng legal_moves

    if is_maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, False)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break # Cắt tỉa Beta (Đối thủ sẽ không bao giờ cho ta nhánh này)
        return max_eval
        
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta_search(board, model, depth - 1, alpha, beta, True)
            board.pop()
            
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Cắt tỉa Alpha (Ta sẽ không bao giờ chọn nhánh này)
        return min_eval

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    global node_count
    node_count = 0 
    eval_cache.clear() # Dọn rác bộ nhớ cho lượt mới
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    best_move = None
    is_ai_white = board.turn == chess.WHITE
    
    # VÒNG LẶP ITERATIVE DEEPENING (Duyệt sâu dần theo ngân sách Node)
    current_depth = 1
    
    print(f"Bắt đầu suy nghĩ. Ngân sách: {MAX_NODES} nodes.")
    
    while node_count < MAX_NODES:
        # Lưu lại best_move của độ sâu trước đó phòng khi bị ngắt giữa chừng
        temp_best_move = legal_moves[0] 
        alpha = -float('inf')
        beta = float('inf')
        
        if is_ai_white:
            best_score = -float('inf')
            for move in legal_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, False)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    temp_best_move = move
                alpha = max(alpha, score)
                
                if node_count >= MAX_NODES: break
        else:
            best_score = float('inf')
            for move in legal_moves:
                board.push(move)
                score = alpha_beta_search(board, model, current_depth - 1, alpha, beta, True)
                board.pop()
                
                if score < best_score:
                    best_score = score
                    temp_best_move = move
                beta = min(beta, score)
                
                if node_count >= MAX_NODES: break
                
        # Nếu đã hoàn thành trọn vẹn 1 tầng độ sâu mà chưa hết ngân sách
        if node_count < MAX_NODES:
            best_move = temp_best_move
            print(f"Hoàn thành Depth {current_depth} | Đã dùng {node_count}/{MAX_NODES} nodes | Tạm chọn: {board.san(best_move)}")
            current_depth += 1
        else:
            print(f"Hết ngân sách khi đang duyệt Depth {current_depth}.")
            if best_move is None:
                best_move = temp_best_move
            
    print(f"=> QUYẾT ĐỊNH: {board.san(best_move)} (Đã duyệt tổng cộng {node_count} trạng thái)")
    return best_move