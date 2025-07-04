from strategies.strategy_base import StrategyBase
import math

class AdvancedCountStrategy(StrategyBase):

    def __init__(self, shoe, logger=None):
        super().__init__()
        self.shoe = shoe
        self.logger = logger
        self.last_true_count = None
        self.initial_bankroll = None

    def true_count(self):
        return self.shoe.true_count()

    def should_play(self):
        return True

    def bet_size(self, min_bet):
        tc = self.true_count()
        bankroll = getattr(self, 'chips', None)

        if bankroll is not None:
            if self.initial_bankroll is None:
                self.initial_bankroll = bankroll
            else:
                dd_now = (self.initial_bankroll - bankroll) / self.initial_bankroll
                if dd_now < 0.05:
                    self.initial_bankroll = bankroll
                    if self.logger:
                        self.logger.debug(f"[AC.bet_size] reset initial_bankroll to {bankroll}")

        if bankroll is not None and self.initial_bankroll is not None:
            dd = (self.initial_bankroll - bankroll) / self.initial_bankroll
            if dd >= 0.20:
                if self.logger:
                    self.logger.debug(f"[AC.bet_size] drawdown {dd*100:.1f}% → min bet")
                return int(min_bet)

        if tc >= 5:
            raw = min_bet * 3.0
        elif tc >= 4:
            raw = min_bet * 2.5
        elif tc >= 3:
            raw = min_bet * 2.0
        elif tc >= 2:
            raw = min_bet * 1.5
        else:
            raw = min_bet * 1.0

        step = min_bet / 2
        amount = round(raw / step) * step

        if bankroll is not None:
            if tc >= 8:
                cap_pct = 0.20
            elif tc >= 5:
                cap_pct = 0.15
            else:
                cap_pct = 0.10

            max_bet = math.floor(bankroll * cap_pct)
            if max_bet < min_bet:
                max_bet = min_bet
            if amount > max_bet:
                if self.logger:
                    self.logger.debug(f"[AC.bet_size] cap {cap_pct*100:.0f}%: {amount} -> {max_bet}")
                amount = max_bet

        if self.logger:
            self.logger.debug(f"[AC.bet_size] TC={tc:.2f}, bet={amount}")
        return int(amount)

    def decide(self, hand, dealer_card):
        tc = self.true_count()
        rules = self.shoe.rules
        bankroll = getattr(self, 'chips', None)
        conservative = False

        if bankroll is not None and self.initial_bankroll is not None:
            dd = (self.initial_bankroll - bankroll) / self.initial_bankroll
            conservative = dd >= 0.20
            if conservative and self.logger:
                self.logger.debug(f"[AC.decide] conservative (DD {dd*100:.1f}%) -> basic")

        if tc < 1 or conservative:
            if self.logger:
                self.logger.debug(f"[AC.decide] basic fallback (TC={tc:.2f}, cons={conservative})")
            return self._basic_decision(hand, dealer_card)

        if self.last_true_count is not None and self.last_true_count < 2 <= tc:
            if self.logger:
                self.logger.debug(f"[AC ALERT] TC z {self.last_true_count:.2f} do {tc:.2f}")
        self.last_true_count = tc

        total = hand.value()
        dealer = dealer_card.value()
        soft = hand.is_soft()
        pair = hand.can_split()

        if rules.allow_surrender and hand.can_surrender() and not soft:
            if total == 14 and dealer == 10 and tc >= 3:
                return "surrender"
            if total == 15 and dealer == 10 and tc >= 0:
                return "surrender"
            if total == 15 and dealer == 9 and tc >= 2:
                return "surrender"
            if total == 15 and dealer == 11 and tc >= 1:
                return "surrender"

        if not soft and not pair:
            if total == 16 and dealer == 10 and tc >= 0:
                return "stand"
            if total == 15 and dealer == 10 and tc >= 4:
                return "stand"
            if total == 10 and dealer == 10 and rules.allow_double and hand.can_double() and tc >= 4:
                return "double"
            if total == 12 and dealer == 3 and tc >= 2:
                return "stand"
            if total == 12 and dealer == 2 and tc >= 3:
                return "stand"
            if total == 11 and dealer == 11 and rules.allow_double and hand.can_double() and tc >= 1:
                return "double"
            if total == 9 and dealer == 2 and rules.allow_double and hand.can_double() and tc >= 1:
                return "double"
            if total == 10 and dealer == 11 and rules.allow_double and hand.can_double() and tc >= 4:
                return "double"
            if total == 9 and dealer == 7 and tc >= 3:
                return "stand"
            if total == 16 and dealer == 9 and tc >= 5:
                return "stand"
            if total == 13 and dealer == 2 and tc >= -1:
                return "stand"
            if total == 12 and dealer == 4 and tc >= 0:
                return "stand"
            if total == 12 and dealer == 5 and tc >= -2:
                return "stand"
            if total == 12 and dealer == 6 and tc >= -1:
                return "stand"
            if total == 13 and dealer == 3 and tc >= -2:
                return "stand"

        if soft:
            if total == 19 and dealer == 6 and rules.allow_double and hand.can_double() and tc >= 1:
                return "double"
            if total == 18 and dealer == 2 and rules.allow_double and hand.can_double() and tc >= 0:
                return "double"
            if total == 18 and dealer == 7 and rules.allow_double and hand.can_double() and tc >= 3:
                return "double"
            if total == 18 and dealer == 8 and tc <= -1:
                return "hit"
            if total == 17 and dealer == 3 and rules.allow_double and hand.can_double() and tc >= 1:
                return "double"
            if total == 16 and dealer == 4 and rules.allow_double and hand.can_double() and tc >= 0:
                return "double"
            if total == 15 and dealer == 4 and rules.allow_double and hand.can_double() and tc >= 0:
                return "double"
            if total == 15 and dealer == 5 and rules.allow_double and hand.can_double() and tc >= 1:
                return "double"

        if dealer == 11 and rules.allow_insurance:
            hand.wants_insurance = (tc >= 3)

        if pair and rules.allow_split:
            rank = hand.cards[0].rank
            if rank in ["A", "8"]:
                return "split"
            if rank == "10" and dealer == 5 and tc >= 5:
                return "split"
            if rank == "10" and dealer == 6 and tc >= 4:
                return "split"
            if rank == "9" and dealer in [2,3,4,5,6,8,9] and tc >= 2:
                return "split"
            if rank == "7" and dealer in range(2,8) and tc >= 3:
                return "split"
            if rank == "6" and dealer in range(2,7) and tc >= 1:
                return "split"
            if rank == "4" and dealer in [5,6] and tc >= 1:
                return "split"
            if rank in ["2", "3"] and dealer in range(2,8) and tc >= 3:
                return "split"

        return self._basic_decision(hand, dealer_card)

    def _basic_decision(self, hand, dealer_card):
        if hand.can_split() and self.shoe.rules.allow_split:
            return self._pair_decision(hand, dealer_card.value())
        if hand.is_soft():
            return self._soft_decision(hand, dealer_card.value())
        return self._hard_decision(hand, dealer_card.value())

    def _pair_decision(self, hand, dealer):
        pair = hand.cards[0].rank
        rules = self.shoe.rules
        if pair in ["A", "8"]:
            return "split"
        if pair in ["2", "3"]:
            return "split" if dealer in range(2,8) else "hit"
        if pair == "4":
            return "split" if dealer in [5,6] else "hit"
        if pair == "5":
            return "double" if dealer in range(2,10) and rules.allow_double and hand.can_double() else "hit"
        if pair == "6":
            return "split" if dealer in range(2,7) else "hit"
        if pair == "7":
            return "split" if dealer in range(2,8) else "hit"
        if pair == "9":
            return "split" if dealer not in [7,10,11] else "stand"
        return "stand"

    def _soft_decision(self, hand, dealer):
        total = hand.value()
        rules = self.shoe.rules
        if total >= 20:
            return "stand"
        if total == 19:
            return "double" if dealer == 6 and rules.allow_double and hand.can_double() else "stand"
        if total == 18:
            if dealer in [2,7,8]:
                return "stand"
            if dealer in range(3,7):
                return "double" if rules.allow_double and hand.can_double() else "stand"
            return "hit"
        if total == 17:
            return "double" if dealer in range(3,7) and rules.allow_double and hand.can_double() else "hit"
        if total in [15,16]:
            return "double" if dealer in range(4,7) and rules.allow_double and hand.can_double() else "hit"
        if total in [13,14]:
            return "double" if dealer in [5,6] and rules.allow_double and hand.can_double() else "hit"
        return "hit"

    def _hard_decision(self, hand, dealer):
        total = hand.value()
        rules = self.shoe.rules
        if total >= 17:
            return "stand"
        if total in [13,14,15,16]:
            return "stand" if dealer in range(2,7) else "hit"
        if total == 12:
            return "stand" if dealer in range(4,7) else "hit"
        if total == 11:
            return "double" if rules.allow_double and hand.can_double() else "hit"
        if total == 10:
            return "double" if dealer in range(2,10) and rules.allow_double and hand.can_double() else "hit"
        if total == 9:
            return "double" if dealer in range(3,7) and rules.allow_double and hand.can_double() else "hit"
        return "hit"

    def want_insurance(self, hand, dealer_card):
        tc = self.true_count()
        return tc >= 3 and self.shoe.rules.allow_insurance
