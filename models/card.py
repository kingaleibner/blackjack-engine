class Card:
    """
    Pojedyncza karta w talii
    """
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def value(self):
        """Zwraca wartość karty w blackjacku"""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11  # as jako 11, korekta w Hand
        return int(self.rank)

    def __str__(self):
        return f"{self.rank} {self.suit}"
