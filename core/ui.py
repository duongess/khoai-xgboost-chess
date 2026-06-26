import chess


def print_board(board: chess.Board, color: str = "white"):
    unicode_pieces = {
        # Trắng (chữ hoa) - ông muốn là ♖ (nhìn đặc/đậm hơn)
        'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
        # Đen (chữ thường) - ông muốn là ♜ (nhìn rỗng/sáng hơn)
        'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙',
        '.': '·'
    }
    
    fen = board.fen().split(' ')[0]
    rows = fen.split('/')
    
    # Nếu chơi quân Đen, lật ngược bàn cờ để hàng 1 lên trên
    if color == "black":
        rows.reverse()
        row_indices = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        # Nếu chơi quân Trắng, hàng 8 ở trên
        row_indices = [8, 7, 6, 5, 4, 3, 2, 1]

    print("\n  a b c d e f g h")
    print("  ---------------")
    
    for idx, row in zip(row_indices, rows):
        line = f"{idx}|"
        for char in row:
            if char.isdigit():
                line += (unicode_pieces['.'] + " ") * int(char)
            else:
                line += unicode_pieces[char] + " "
        print(line + f"|{idx}")
        
    print("  ---------------")
    print("  a b c d e f g h\n")