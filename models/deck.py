import random
from .card import Card

class Deck:
    """
    Talia kart
    """
    suits = ['Pik', 'Kier', 'Karo', 'Trefl']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.cards = []
        self._create_deck()
        self.shuffle()

    def _create_deck(self):
        self.cards = [Card(rank, suit)
                      for _ in range(self.num_decks)
                      for suit in self.suits
                      for rank in self.ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            self._create_deck()
            self.shuffle()
        return self.cards.pop()

    def cards_left(self):
        return len(self.cards)
