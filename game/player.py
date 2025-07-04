from models.hand import Hand
from models.chips import Chips
from models.deck import Deck
import logging

class Player:
    """Reprezentuje gracza przy stole"""

    def __init__(self, name, chips, strategy):
        self.name = name
        self.chips = chips if isinstance(chips, Chips) else Chips(chips)
        self.strategy = strategy
        self.hands = []
        self.active_bets = []
        self.insurance_bets = {} 

    def new_round(self):
        self.hands = [Hand()]
        self.active_bets = []

    def place_bet(self, amount):
        try:
            used_chips = self.chips.bet(amount)
        except ValueError:
            if self.chips.make_change(amount):
                try:
                    used_chips = self.chips.bet(amount)
                except ValueError:
                    raise ValueError(f"{self.name} nie może postawić zakładu: brak odpowiednich żetonów.")
            else:
                raise ValueError(f"{self.name} nie może postawić zakładu: za mało środków.")

        self.active_bets.append(amount)
        return used_chips

    def win(self, hand_index: int, multiplier: float = 2.0, blackjack=False):
        if hand_index >= len(self.active_bets):
            return

        bet = self.active_bets[hand_index]
        multiplier = 1.5 if blackjack else 1.0
        payout = int(bet * (1 + multiplier)) 
        self.chips.payout(payout)
        logging.info(f"[WIN_FN] {self.name} hand {hand_index}, bet={bet}, payout={payout}, blackjack={blackjack}")

    def lose(self, hand_index: int):
        if hand_index >= len(self.active_bets):
            return

        bet = self.active_bets[hand_index]

    def push(self, hand_index: int):
        if hand_index >= len(self.active_bets):
            return

        bet = self.active_bets[hand_index]
        self.chips.payout(bet)


    def can_continue(self, min_bet):
        return self.chips.total() >= min_bet
    
    def split_hand(self, hand_index: int, deck: Deck):
        if hand_index >= len(self.hands):
            return

        hand = self.hands[hand_index]

        if not hand.can_split():
            return

        card_to_move = hand.cards.pop()
        new_hand = Hand()
        new_hand.add_card(card_to_move)

        hand.add_card(deck.draw())
        new_hand.add_card(deck.draw())

        original_bet = self.active_bets[hand_index]

        try:
            self.place_bet(original_bet)
        except ValueError:
            return

        self.hands.append(new_hand)

    def double_down(self, hand_index: int, deck: Deck):
        if hand_index >= len(self.hands):
            return

        hand = self.hands[hand_index]
        if not hand.is_active:
            return

        if len(hand.cards) != 2:
            return

        current_bet = self.active_bets[hand_index]
        if self.chips.total() < current_bet:
            return

        self.place_bet(current_bet) 
        self.active_bets[hand_index] *= 2
        hand.add_card(deck.draw()) 
        hand.is_active = False 

    def surrender_hand(self, hand_index: int):
        hand = self.hands[hand_index]
        if not hand.is_active:
            return

        current_bet = self.active_bets[hand_index]
        refund = current_bet // 2
        self.chips.payout(refund)
        self.active_bets[hand_index] = 0

        hand.is_active = False

    def take_insurance(self, hand_index: int):
        current_bet = self.active_bets[hand_index]
        insurance_bet = current_bet // 2

        if self.chips.total() < insurance_bet:
            return

        self.chips.bet(insurance_bet)
        self.insurance_bets[hand_index] = insurance_bet

    def __str__(self):
        return f"{self.name} (żetony: {self.chips.total()})"
