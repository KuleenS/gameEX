import pickle

import random 

import chess

from stockfish import Stockfish

from tqdm import tqdm

stockfish = Stockfish(path="Stockfish/src/stockfish")

dataset = []

def generate_random_board():
    board = chess.Board()

    random_moves = random.randint(1, 80)

    random_moves_for_board = []

    for i in range(random_moves):
        moves = list(board.legal_moves)

        random_move = random.choice(moves)
        board.push(random_move)

        random_moves_for_board.append(random_move)
    
    return board, random_moves_for_board

with tqdm(total=1000) as pbar:
    while len(dataset) < 1000:
        try: 
            board, random_moves_for_board = generate_random_board()

            stockfish.set_position(random_moves_for_board)

            legal_moves = list(board.legal_moves)

            if len(legal_moves) == 0:
                raise IndexError

            board_visual = stockfish.get_board_visual()

            moves_to_get = board.fen()

            top_moves = stockfish.get_top_moves(len(legal_moves))

            dataset.append({"board": board_visual, "legal_moves" : legal_moves, "moves_to_get": moves_to_get, "top_moves": top_moves})

            pbar.update(1)
        except IndexError:
            print("Board Failed, retrying...")
            continue

with open("dataset_small.pkl", "wb") as f:
    pickle.dump(dataset, f)
