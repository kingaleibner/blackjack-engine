import os
import csv


class ExpertLogger:
    def __init__(self, filename="expert_data.csv"):
        self.filename = filename
        self.rows = []

        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "player_total", "dealer_upcard", "is_soft", "true_count",
                    "can_split", "can_double", "num_cards",
                    "dealer_has_ace", "player_has_ace", "player_blackjack",
                    "hand_difference", "true_count_bucket", "seq",
                    "action", "reward", "outcome"
                ])

    def bucketize_count(self, count):
        if count < 0:
            return "low"
        elif count < 2:
            return "mid"
        else:
            return "high"

    def log(self, hand, dealer_card, true_count, action, reward, outcome, seq=1):
        dealer_val = min(dealer_card.value(), 10)
        row = [
            hand.value(),
            dealer_val,
            hand.is_soft(),
            true_count,
            hand.can_split(),
            hand.can_double(),
            len(hand.cards),
            dealer_card.rank == 'A',
            any(c.rank == 'A' for c in hand.cards),
            hand.is_blackjack(),
            hand.value() - dealer_val,
            self.bucketize_count(true_count),
            seq,
            action,
            reward,
            outcome
        ]
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        with open(self.filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
        self.rows = []
