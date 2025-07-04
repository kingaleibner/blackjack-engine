from models.hand import Hand
from game.casino_rules import CasinoRules

class Dealer:
    """Reprezentuje krupiera"""

    def __init__(self, rules: CasinoRules = CasinoRules.classic()):
        self.hand = Hand()
        self.rules = rules

    def new_round(self):
        self.hand = Hand()

    def play(self, deck):
        while True:
            value = self.hand.value()
            if value < 17:
                self.hand.add_card(deck.draw())
            elif value == 17 and self.rules.hit_soft_17 and self._has_soft_17():
                self.hand.add_card(deck.draw())
            else:
                break

    def _has_soft_17(self):
        total = sum(card.value() for card in self.hand.cards)
        num_aces = sum(1 for card in self.hand.cards if card.rank == 'A')
        return total == 17 and num_aces > 0

    def upcard(self):
        return self.hand.cards[0] if self.hand.cards else None

    def __str__(self):
        return f"Krupier: {self.hand}"
