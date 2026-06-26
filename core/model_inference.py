import chess
import numpy as np
import xgboost as xgb
from training.data_pipeline import board_to_matrix

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    # Không cần load model ở đây nữa
    legal_moves = list(board.legal_moves)
    
    best_move = None
    best_score = -float('inf') 
    
    # 3. Duyệt qua từng nước, "thử" đi và hỏi AI xem thế cờ đó mạnh hay yếu
    for move in legal_moves:
        board.push(move)
        
        # Chụp X-quang thế cờ sau khi đã đi
        matrix = board_to_matrix(board)
        
        # AI chấm điểm (score càng cao = thế cờ càng lợi cho quân mình)
        score = model.predict(matrix.reshape(1, -1))[0]
        
        # Tìm nước có điểm cao nhất
        if score > best_score:
            best_score = score
            best_move = move
            
        board.pop() # Hoàn tác (undo) để thử nước tiếp theo
    
    print(f"AI chon nuoc di: {board.san(best_move)}")
        
    return best_move