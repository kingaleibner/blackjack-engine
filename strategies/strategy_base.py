from abc import ABC, abstractmethod

class StrategyBase(ABC):
    """
    Bazowa klasa strategii do dziedziczenia
    """

    def __init__(self, min_bet=10):
        self.min_bet = min_bet

    @abstractmethod
    def decide(self, hand, dealer_card):
        """
        Decyzja gracza na podstawie aktualnej ręki i karty krupiera
        Zwraca: "hit", "stand", "double", "split"
        """
        pass

    def get_bet_amount(self, player, table):
        """
        Zwraca kwotę zakładu na podstawie stanu gracza i stołu.
        Domyślnie zwraca minimalny zakład.
        """
        return self.min_bet

    def update_count(self, cards):
        """
        (Aktualizuje stan strategii
        """
        pass

    def reset(self):
        """
        Resetuje stan strategii
        """
        pass

    def want_insurance(self, hand, dealer_card):
        return False
