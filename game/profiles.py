import random

class PlayerProfile:
    def __init__(self, style: str, name: str = None):
        self.name = name
        self.style = style
        self.bet_history = []

    def decide_bet(self, true_count, min_bet=10):
        if self.style == 'aggressive':
            if true_count >= 3:
                bet = min_bet * 4
            elif true_count >= 2:
                bet = min_bet * 3
            elif true_count >= 1:
                bet = min_bet * 2
            else:
                bet = min_bet
        elif self.style == 'cautious':
            if true_count >= 3:
                bet = min_bet * 2
            elif true_count >= 2:
                bet = min_bet * 1.5
            else:
                bet = min_bet
        elif self.style == 'hilo':
            if true_count >= 3:
                bet = min_bet * 3
            elif true_count >= 2:
                bet = min_bet * 2
            elif true_count >= 1:
                bet = min_bet * 1.5
            else:
                bet = min_bet
        else: 
            bet = random.choice([min_bet, min_bet * 2, min_bet * 3])
        self.bet_history.append(bet)
        return bet
