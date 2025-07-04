from strategies.strategy_base import StrategyBase
import math
import random


class RandomProfileStrategy(StrategyBase):

    def __init__(self, shoe, profile, logger=None):
        super().__init__()
        self.shoe = shoe
        self.profile = profile
        self.logger = logger
        from strategies.basic import BasicStrategy
        self.basic = BasicStrategy(shoe, logger=logger)

    def get_bet_amount(self, player, table):
        tc = self.shoe.true_count()
        min_bet = table.min_bet
        bet = self.profile.decide_bet(tc, min_bet)
        if self.logger:
            self.logger.debug(f"[RandomProfile.bet] TC={tc:.2f}, style={self.profile.style}, bet={bet}")
        return int(bet)

    def decide(self, hand, dealer_card):
        return self.basic.decide(hand, dealer_card)

    def want_insurance(self, hand, dealer_card):
        return False
    