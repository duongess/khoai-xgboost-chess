import chess
import numpy as np
import xgboost as xgb

# Đổi từ board_to_matrix sang extract_features
from core.utils import extract_features 

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    batch_matrices = []
    
    # 1. Trích xuất đặc trưng cho TẤT CẢ các nước đi
    for move in legal_moves:
        board.push(move)
        
        # Gọi hàm mới với 69 features
        features = extract_features(board) 
        batch_matrices.append(features)
        
        board.pop() 
        
    # Chuyển đổi thành ma trận 2 chiều (Số_nước_đi, 69)
    X_batch = np.array(batch_matrices)
    
    # 2. Gọi AI dự đoán toàn bộ cùng 1 lúc (Nhanh hơn rất nhiều so với gọi trong vòng lặp)
    # Lưu ý: Nếu bạn đã chuyển sang XGBClassifier, hãy cân nhắc dùng model.predict_proba(X_batch)[:, 1]
    ai_scores = model.predict(X_batch)
    
    # 3. Tìm nước đi có điểm số cao nhất
    best_move = None
    best_score = -float('inf') 
    
    for i, move in enumerate(legal_moves):
        score = ai_scores[i]
        if score > best_score:
            best_score = score
            best_move = move
            
    print(f"AI chon nuoc di: {board.san(best_move)} voi muc diem {best_score:.4f}")
        
    return best_move