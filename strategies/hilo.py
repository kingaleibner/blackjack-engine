from strategies.strategy_base import StrategyBase
import math

class HiLoStrategy(StrategyBase):

    def __init__(self, shoe, logger=None):
        super().__init__()
        self.shoe = shoe
        self.logger = logger
        from strategies.basic import BasicStrategy
        self.basic = BasicStrategy(shoe, logger=logger)

    def bet_size(self, min_bet):
        tc = self.shoe.true_count()
        if tc >= 3:
            raw = min_bet * 3.0
        elif tc >= 2:
            raw = min_bet * 2.0
        elif tc >= 1:
            raw = min_bet * 1.5
        else:
            raw = min_bet
        step = min_bet / 2
        amount = round(raw / step) * step
        if self.logger:
            self.logger.debug(f"[HiLoBasic.bet_size] TC={tc:.2f}, raw={raw:.1f} -> amount={amount}")
        return int(amount)

    def decide(self, hand, dealer_card):
        return self.basic.decide(hand, dealer_card)

    def want_insurance(self, hand, dealer_card):
        return False
