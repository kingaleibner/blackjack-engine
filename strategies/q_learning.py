import joblib
import numpy as np
from .strategy_base import StrategyBase
from game.profiles import PlayerProfile

class QLearningStrategy(StrategyBase):

    def __init__(self, shoe, profile: PlayerProfile, q_table_path: str,
                 logger=None):
        """
        :param shoe: obiekt Shoe z metodą true_count()
        :param profile: instancja PlayerProfile
        :param q_table_path: ścieżka do pliku q_table.pkl
        :param logger: opcjonalny logger
        """
        super().__init__()
        self.shoe    = shoe
        self.profile = profile
        self.logger  = logger

        self.q_table: dict = joblib.load(q_table_path)

        self.actions = ["hit", "stand", "double", "split", "surrender"]

    def _discretize(self, true_count: float, player_total: int,
                     dealer_up: int, is_soft: bool,
                     num_cards: int, can_split: bool):
        total_b  = min(max(int(player_total), 4), 21)
        dealer_b = min(max(int(dealer_up), 2), 11)
        tc_b     = min(max(int(round(true_count)), -10), 10)
        soft_b   = 1 if is_soft else 0
        nc_b     = 0 if int(num_cards) == 2 else 1
        split_b  = 1 if can_split else 0
        return (tc_b, total_b, dealer_b, soft_b, nc_b, split_b)

    def decide(self, hand, dealer_card) -> str:
        tc = self.shoe.true_count()
        pt = hand.value()
        du = dealer_card.value()
        is_soft   = hand.is_soft()
        num_cards = len(hand.cards)
        can_split = hand.can_split() and hasattr(self.shoe, 'rules') and self.shoe.rules.allow_split

        state = self._discretize(tc, pt, du, is_soft, num_cards, can_split)

        q_dict = self.q_table.get(state, {})
        if q_dict:
            action = max(q_dict.items(), key=lambda x: x[1])[0]
        else:
            action = "stand"

        if self.logger:
            self.logger.debug(f"[QLearn] Stan={state} -> {action}")

        return action

    def bet_size(self, min_bet: int) -> int:
        tc = self.shoe.true_count()
        bet = self.profile.decide_bet(tc, min_bet)
        if self.logger:
            self.logger.debug(f"[QLearn.bet] TC={tc:.2f} -> {bet}")
        return bet

    def want_insurance(self, hand, dealer_card) -> bool:
        return False
