import chess
import math
import numpy as np
from engine.encoder import board_to_tensor
import torch

C_PUCT = 1.25

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visits = 0
        self.value = 0
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self):
        if self.visits == 0:
            return float('inf')
        q = self.value / self.visits
        u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

    def expand(self, policy):
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            child_board = self.board.copy()
            child_board.push(move)
            prior = policy[move.to_square]
            child = MCTSNode(child_board, parent=self, move=move, prior=prior)
            self.children.append(child)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb_score())

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)


def mcts_search(board, model, device, simulations=100):
    root = MCTSNode(board)

    for _ in range(simulations):
        node = root

        while not node.is_leaf() and not node.board.is_game_over():
            node = node.best_child()

        if not node.board.is_game_over():
            tensor = board_to_tensor(node.board)
            x = torch.tensor(tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = model(x)
            policy = torch.softmax(policy, dim=1).squeeze().cpu().numpy()
            value = float(value.squeeze())
            node.expand(policy)
        else:
            result = node.board.result()
            if result == "1-0": value = 1.0
            elif result == "0-1": value = -1.0
            else: value = 0.0

        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    if not root.children:
        return list(board.legal_moves)[0]
    return root.most_visited_child().move


if __name__ == "__main__":
    from model.network import ChessNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    model.load_state_dict(torch.load("model/chess_model.pt", map_location=device))
    model.eval()

    board = chess.Board()
    print("Thinking with MCTS...")
    move = mcts_search(board, model, device, simulations=50)
    print(f"MCTS best move: {move}")

'''python -m engine.mcts'''