from shutil import move

from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QInputDialog, QLabel, QWidget, QGridLayout, QVBoxLayout
from PySide6.QtCore import Qt, QTimer
import chess
from config.Setting import SIZE_SQUARE, get_chess_pieces, get_model_path
import xgboost as xgb
from core.model_inference import get_ai_move

class BoardWidget(QWidget):
    def __init__(self, board, model_name):
        super().__init__()
        self.setFixedSize(SIZE_SQUARE * 8, SIZE_SQUARE * 8)
        self.board = board
        self.layout = QGridLayout(self)
        self.layout.setSpacing(0)
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(get_model_path(model_name)))
        self.draw_board()
    
    def draw_board(self):
        king_square = None
        if self.board.is_check():
            # Tìm vị trí ô chứa Vua đang bị chiếu
            king_square = self.board.king(self.board.turn)

        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget: widget.setParent(None)
            
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = self.board.piece_at(square)
                
                square_widget = QWidget()
                base_color = "#eeeed2" if (rank + file) % 2 == 0 else "#769656"
                
                if square == self.selected_square:
                    bg_color = "#FFFF33"
                elif square == king_square:
                    bg_color = "#ff4d4d" 
                else:
                    bg_color = base_color
                square_widget.setStyleSheet(f"background-color: {bg_color};")
                cell_layout = QGridLayout(square_widget)
                cell_layout.setContentsMargins(2, 2, 2, 2)

                if file == 0:
                    lbl = QLabel(str(8 - rank))
                    lbl.setStyleSheet("color: #000; font-weight: bold;")
                    cell_layout.addWidget(lbl, 0, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
                
                if rank == 7:
                    lbl = QLabel(chr(97 + file))
                    lbl.setStyleSheet("color: #000; font-weight: bold;")
                    cell_layout.addWidget(lbl, 0, 0, alignment=Qt.AlignBottom | Qt.AlignRight)

                if piece:
                    svg = QSvgWidget(str(get_chess_pieces(piece.symbol())))
                    svg.setFixedSize(SIZE_SQUARE - 30, SIZE_SQUARE - 30)
                    cell_layout.addWidget(svg, 0, 0, alignment=Qt.AlignCenter)
                    if square in self.legal_moves_for_selected:
                        square_widget.setStyleSheet("background-color: #ff4d4d;")
                
                elif square in self.legal_moves_for_selected:
                    dot = QLabel("●")
                    dot.setStyleSheet("color: rgba(0,0,0,0.2); font-size: 30px;")
                    cell_layout.addWidget(dot, 0, 0, alignment=Qt.AlignCenter)

                self.layout.addWidget(square_widget, rank, file)

    def mousePressEvent(self, event):
        file = int(event.position().x() // SIZE_SQUARE)
        rank = 7 - int(event.position().y() // SIZE_SQUARE)
        if not (0 <= file <= 7 and 0 <= rank <= 7): return
        
        clicked_sq = chess.square(file, rank)
        
        # Nếu đã chọn quân
        if self.selected_square is not None:
            # Lấy tất cả nước đi hợp lệ từ ô đang chọn
            possible_moves = [m for m in self.board.legal_moves if m.from_square == self.selected_square]
            
            move = None
            for m in possible_moves:
                if m.to_square == clicked_sq:
                    move = m 
                    break
            
            if move:
                if move.promotion is not None:
                    print("Đang gọi Dialog phong cấp...") 
                    self.show_promotion_dialog(move)
                    return
                
                # Nước đi thường
                self.board.push(move)
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.draw_board()
                
                if self.check_game_over(): return 
                QTimer.singleShot(200, self.ai_move)
                return
            
            # Nếu click vào quân của chính mình thì đổi chọn
            elif self.board.piece_at(clicked_sq) and self.board.piece_at(clicked_sq).color == self.board.turn:
                self.selected_square = clicked_sq
                self.legal_moves_for_selected = [m.to_square for m in self.board.legal_moves if m.from_square == clicked_sq]
            else:
                self.selected_square = None
                self.legal_moves_for_selected = []
        else:
            # Chọn quân
            if self.board.piece_at(clicked_sq) and self.board.piece_at(clicked_sq).color == self.board.turn:
                self.selected_square = clicked_sq
                self.legal_moves_for_selected = [m.to_square for m in self.board.legal_moves if m.from_square == clicked_sq]
        
        self.draw_board()
    
    def show_promotion_dialog(self, move):
        items = ["Queen", "Rook", "Bishop", "Knight"]
        item, ok = QInputDialog.getItem(self, "Phong cấp", "Chọn quân:", items, 0, False)
        
        if ok and item:
            promo = {"Queen": chess.QUEEN, "Rook": chess.ROOK, "Bishop": chess.BISHOP, "Knight": chess.KNIGHT}[item]
            move.promotion = promo
            self.board.push(move)
            self.draw_board()
            
            # GỌI AI SAU KHI PHONG CẤP
            if not self.check_game_over():
                QTimer.singleShot(200, self.ai_move)
        
    def ai_move(self):
        move = get_ai_move(self.board, self.model)
        if move:
            self.board.push(move)
            self.draw_board()
            self.check_game_over()

    def check_game_over(self):
        if self.board.is_game_over():
            self.close()
            return True
        return False