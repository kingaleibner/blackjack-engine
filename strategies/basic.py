from .strategy_base import StrategyBase

class BasicStrategy(StrategyBase):
    def __init__(self, shoe, logger=None):
        super().__init__()
        self.logger = logger
        self.shoe = shoe

    def decide(self, hand, dealer_card):
        player_total = hand.value()
        dealer_value = dealer_card.value()
        is_soft = hand.is_soft()
        can_split = hand.can_split()
        rules = self.shoe.rules

        if self.logger:
            self.logger.debug(
                f"[BS] Hand: {hand}, Dealer: {dealer_card.rank} ({dealer_value}), "
                f"Total: {player_total}, Soft: {is_soft}, Split: {can_split}"
            )

        if can_split:
            action = self._split_decision(hand.cards[0].rank, dealer_value)
        elif hand.can_surrender() and not is_soft and rules.allow_surrender and (
            (player_total == 16 and dealer_value in [9, 10, 11]) or
            (player_total == 15 and dealer_value == 10)
        ):
            action = 'surrender'
        elif is_soft:
            action = self._soft_total_decision(player_total, dealer_value, hand)
        else:
            action = self._hard_total_decision(player_total, dealer_value, hand)

        valid_actions = {"hit", "stand", "double", "split", "surrender"}
        if action not in valid_actions:
            if self.logger:
                self.logger.warning(
                    f"[BS] [FATAL] Nieznana akcja: {action} — fallback na STAND"
                )
            return "stand"
        return action

    def _split_decision(self, pair, dealer_value):
        rules = self.shoe.rules
        if pair in ['A', '8']:
            return 'split'
        if pair in ['5', '10']:
            return 'double' if pair == '5' and dealer_value in range(2, 10) else 'hit'
        if pair in ['2', '3']:
            if dealer_value in [4, 5, 6, 7] or (
                rules.allow_double_after_split and dealer_value in [2, 3]
            ):
                return 'split'
            return 'hit'
        if pair == '4':
            if rules.allow_double_after_split and dealer_value in [5, 6]:
                return 'split'
            return 'hit'
        if pair == '6':
            if dealer_value in [3, 4, 5, 6] or (
                rules.allow_double_after_split and dealer_value == 2
            ):
                return 'split'
            return 'hit'
        if pair == '7':
            return 'split' if dealer_value in range(2, 8) else 'hit'
        if pair == '9':
            return 'split' if dealer_value not in [7, 10, 11] else 'stand'
        return 'stand'

    def _soft_total_decision(self, total, dealer_value, hand):
        if total >= 20:
            return 'stand'
        if total == 19:
            return self._try_double_or_stand(hand, dealer_value == 6)
        if total == 18:
            if dealer_value in range(3, 7):
                return self._try_double_or_stand(hand, True)
            if dealer_value in [2, 7, 8]:
                return 'stand'
            return 'hit'
        if total == 17:
            return self._try_double_or_hit(hand, dealer_value in range(3, 7))
        if total in [15, 16]:
            return self._try_double_or_hit(hand, dealer_value in range(4, 7))
        if total in [13, 14]:
            return self._try_double_or_hit(hand, dealer_value in [5, 6])
        return 'hit'

    def _hard_total_decision(self, total, dealer_value, hand):
        if total >= 17:
            return 'stand'
        if total in [13, 14, 15, 16]:
            return 'stand' if dealer_value in range(2, 7) else 'hit'
        if total == 12:
            return 'stand' if dealer_value in range(4, 7) else 'hit'
        if total == 11:
            return self._try_double_or_hit(hand, dealer_value != 11)
        if total == 10:
            return self._try_double_or_hit(hand, dealer_value in range(2, 10))
        if total == 9:
            return self._try_double_or_hit(hand, dealer_value in range(3, 7))
        return 'hit'

    def _try_double_or_hit(self, hand, condition: bool):
        if condition and hand.can_double():
            return 'double'
        if self.logger:
            self.logger.debug(
                "[BS] Zakazane double. Fallback na HIT."
            )
        return 'hit'

    def _try_double_or_stand(self, hand, condition: bool):
        if condition and hand.can_double():
            return 'double'
        if self.logger:
            self.logger.debug(
                "[BS] Zakazane double. Fallback na STAND."
            )
        return 'stand'

    def want_insurance(self, hand, dealer_card):
        return False
