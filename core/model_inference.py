import chess
import numpy as np
import xgboost as xgb
from training.data_pipeline import board_to_matrix

def evaluate_intuition(board: chess.Board) -> float:
    # 0. ĐIỀU KIỆN TỐI THƯỢNG (Thắng/Thua/Hòa)
    if board.is_checkmate():
        # Người vừa đi đã chiếu bí đối phương -> Trả về điểm cực đại
        return 99999.0 
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0 # Hòa cờ không có lợi thế

    # Phân định phe ta (người vừa thực hiện nước đi) và phe địch
    my_color = not board.turn
    opp_color = board.turn

    my_score = 0.0
    opp_score = 0.0

    # =======================================================
    # 1. THIỆT QUÂN (Tương đối) - Điểm nền tảng
    # =======================================================
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900
    }
    for pt, val in PIECE_VALUES.items():
        my_score += len(board.pieces(pt, my_color)) * val
        opp_score += len(board.pieces(pt, opp_color)) * val

    # Lấy vị trí Vua để tính điểm Hở vua và Thế công
    my_king_sq = board.king(my_color)
    opp_king_sq = board.king(opp_color)

    # Duyệt 64 ô cờ để phân tích chi tiết không gian và cấu trúc
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue

        color = piece.color
        pt = piece.piece_type
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        # Số ô mà quân này đang kiểm soát (tính toán không gian)
        mobility = len(board.attacks(sq))

        # =======================================================
        # 2. QUÂN PHẾ VẬT (Ít) & 3. KHÔNG GIAN (Tương đối)
        # =======================================================
        # Thưởng không gian: Mỗi ô kiểm soát được cộng 2 điểm
        space_bonus = mobility * 2
        
        # Phạt quân phế vật: Quân nhẹ/nặng bị kẹt (ít đường đi)
        bad_piece_penalty = 0
        if pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK] and mobility <= 2:
            bad_piece_penalty = 15 # Trừ 15 điểm nếu bị kẹt

        # Kiểm soát trung tâm (cực kỳ quan trọng với trực giác con người)
        center_bonus = 0
        if pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP]:
            if sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
                center_bonus = 30
            elif sq in [chess.C3, chess.D3, chess.E3, chess.F3, chess.C6, chess.D6, chess.E6, chess.F6]:
                center_bonus = 10

        if color == my_color:
            my_score += (space_bonus + center_bonus - bad_piece_penalty)
        else:
            opp_score += (space_bonus + center_bonus - bad_piece_penalty)

        # =======================================================
        # 6. TỐT THÔNG (Nhiều)
        # =======================================================
        if pt == chess.PAWN:
            is_passed = True
            # Hướng tiến của Tốt
            front_ranks = range(rank_idx + 1, 8) if color == chess.WHITE else range(0, rank_idx)
            adj_files = [file_idx - 1, file_idx, file_idx + 1]
            
            for r in front_ranks:
                for f in adj_files:
                    if 0 <= f <= 7:
                        check_sq = chess.square(f, r)
                        check_p = board.piece_at(check_sq)
                        # Nếu có Tốt đối phương ngáng đường hoặc ở cột bên cạnh
                        if check_p and check_p.piece_type == chess.PAWN and check_p.color != color:
                            is_passed = False
                            break
                if not is_passed: break
            
            if is_passed:
                # Tốt thông càng tiến sát phong cấp, điểm càng KHỦNG (Nhiều)
                bonus = 50 + (rank_idx if color == chess.WHITE else 7 - rank_idx) * 30
                if color == my_color: my_score += bonus
                else: opp_score += bonus

    # =======================================================
    # 4. THẾ CÔNG (Nhiều)
    # =======================================================
    # Nước đi tạo ra thế Chiếu Vua (Check) -> Áp lực cực lớn
    if board.is_check():
        my_score += 80 

    # Đếm số quân phe ta đang uy hiếp trực tiếp vào khu vực quanh Vua địch
    opp_king_zone = board.attacks(opp_king_sq) if opp_king_sq else []
    for zone_sq in opp_king_zone:
        if board.is_attacked_by(my_color, zone_sq):
            my_score += 20 # Nhắm 1 tia vào Vua địch -> +20 điểm

    my_king_zone = board.attacks(my_king_sq) if my_king_sq else []
    for zone_sq in my_king_zone:
        if board.is_attacked_by(opp_color, zone_sq):
            opp_score += 20

    # =======================================================
    # 5. HỞ VUA (Nhiều)
    # =======================================================
    def evaluate_king_safety(king_sq, is_white):
        if not king_sq: return 0
        safety_score = 0
        k_file = chess.square_file(king_sq)
        k_rank = chess.square_rank(king_sq)
        
        # Chỉ xét Hở vua nếu Vua vẫn đang ở hàng dưới cùng (chưa vào tàn cuộc)
        if (is_white and k_rank <= 1) or (not is_white and k_rank >= 6):
            shield_rank = k_rank + 1 if is_white else k_rank - 1
            shield_count = 0
            for f in [k_file - 1, k_file, k_file + 1]:
                if 0 <= f <= 7:
                    shield_sq = chess.square(f, shield_rank)
                    p = board.piece_at(shield_sq)
                    if p and p.piece_type == chess.PAWN and p.color == is_white:
                        shield_count += 1
            
            # Khuyến khích có 2-3 Tốt che chắn. Phạt nặng nếu trống hoác
            if shield_count == 0: safety_score -= 80
            elif shield_count == 1: safety_score -= 30
            else: safety_score += 20
        else:
            # Vua lang thang ở giữa bàn cờ trong trung cuộc bị phạt rất nặng
            safety_score -= 100 
            
        return safety_score

    my_score += evaluate_king_safety(my_king_sq, my_color == chess.WHITE)
    opp_score += evaluate_king_safety(opp_king_sq, opp_color == chess.WHITE)

    # TRẢ VỀ ĐIỂM CHÊNH LỆCH: Tốt cho người vừa đi thì trả ra số DƯƠNG LỚN
    return my_score - opp_score

def get_ai_move(board: chess.Board, model: xgb.XGBRegressor) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    batch_matrices = []
    intuition_scores = []
    
    # Bước 1: Chuẩn bị dữ liệu và chấm điểm trực giác cho TẤT CẢ nước đi
    for move in legal_moves:
        board.push(move)
        
        # 1.1 Lấy điểm trực giác (Vật chất, an toàn Vua...)
        intuition_scores.append(evaluate_intuition(board))
        
        # 1.2 Trích xuất ma trận cho AI
        batch_matrices.append(board_to_matrix(board))
        
        board.pop()
        
    # Bước 2: AI chấm điểm phong cách cho toàn bộ nước đi cùng lúc
    X_batch = np.array(batch_matrices)
    ai_scores = model.predict(X_batch) 
    
    # Bước 3: Kết hợp điểm số (Công thức Hybrid)
    best_move = None
    best_total_score = -float('inf')
    
    # Trọng số có thể tinh chỉnh: 
    # W1 càng cao -> Đánh càng thực dụng (chính xác vật chất)
    # W2 càng cao -> Đánh càng giống con người/kiện tướng (phong cách)
    WEIGHT_INTUITION = 1.0 
    WEIGHT_STYLE = 5.0 # Ví dụ: AI trả về điểm nhỏ từ 0-1 nên cần nhân hệ số lớn
    
    for i, move in enumerate(legal_moves):
        total_score = (WEIGHT_INTUITION * intuition_scores[i]) + (WEIGHT_STYLE * ai_scores[i])
        
        if total_score > best_total_score:
            best_total_score = total_score
            best_move = move
            
    # Hiển thị comment theo yêu cầu
    print(f"Nuoc di duoc chon: {board.san(best_move)} (Diem tong: {best_total_score:.2f})")
    
    return best_move