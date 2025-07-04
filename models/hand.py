import copy

class Hand:
    """
    Ręka gracza lub krupiera
    """
    def __init__(self):
        self.cards = []
        self.is_active = True
        self.insurance = 0

    def add_card(self, card):
        self.cards.append(card)

    def value(self):
        total = sum(card.value() for card in self.cards)
        num_aces = sum(1 for card in self.cards if card.rank == 'A')

        while total > 21 and num_aces:
            total -= 10
            num_aces -= 1
        return total
    
    def clone(self):
        return copy.deepcopy(self)

    def is_blackjack(self):
        return len(self.cards) == 2 and self.value() == 21

    def is_busted(self):
        return self.value() > 21
    
    def is_soft(self):
        if self.is_blackjack():
            return False
        return self.value() != self.hard_value()
    
    def hard_value(self):
        total = 0
        for card in self.cards:
            if card.rank == 'A':
                total += 1
            else:
                total += card.value()
        return total

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank
    
    def can_double(self):
        return len(self.cards) == 2

    def can_surrender(self):
        return len(self.cards) == 2 and self.is_active and not getattr(self, 'has_moved', False)
    
    def __str__(self):
        return ', '.join(str(card) for card in self.cards) + f" (wartość: {self.value()})"
