import random
import copy
from models.deck import Deck
from game.casino_rules import CasinoRules

class Shoe:
    def __init__(self, rules: CasinoRules = CasinoRules.classic()):
        self.rules = rules
        self.num_decks = rules.num_decks
        self.cut_card_position = rules.cut_card_position
        self.cards = []
        self.cut_card_reached = False
        self.running_count = 0
        self.shuffle()

    def shuffle(self):
        self.cards = []
        for _ in range(self.rules.num_decks):
            deck = Deck()
            self.cards.extend(deck.cards)
        random.shuffle(self.cards)
        self.running_count = 0 
        self.cut_card_reached = False

    def count_card(self, card):
        if card.rank in ['2', '3', '4', '5', '6']:
            self.running_count += 1
        elif card.rank in ['10', 'J', 'Q', 'K', 'A']:
            self.running_count -= 1

    def observe_cards(self, cards):
        for card in cards:
            self.count_card(card)

    def draw(self):
        if not self.cut_card_reached and \
        len(self.cards) < self.num_decks * 52 * (1 - self.cut_card_position):
            self.cut_card_reached = True

        if self.cut_card_reached or len(self.cards) == 0:
            self.reshuffle()
            self.cut_card_reached = False

        card = self.cards.pop()
        self.count_card(card)
        return card

    def cards_left(self):
        return len(self.cards)

    def decks_left(self):
        decks = self.cards_left() / 52.0
        return decks if decks > 0 else 0.01

    def true_count(self):
        return self.running_count / self.decks_left()

    def needs_reshuffle(self):
        return self.cut_card_reached

    def reshuffle(self):
        self.shuffle()

    def clone(self):
        return copy.deepcopy(self)
