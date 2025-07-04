class CasinoRules:
    def __init__(self,
                 num_decks=4,
                 cut_card_position=0.7,
                 blackjack_payout=(3, 2),
                 allow_double=True,
                 allow_double_after_split=True,
                 allow_split=True,
                 allow_resplit_aces=False,
                 allow_surrender=True,
                 hit_soft_17=True,
                 allow_insurance=True):
        self.num_decks = num_decks
        self.cut_card_position = cut_card_position
        self.blackjack_payout = blackjack_payout
        self.allow_double = allow_double
        self.allow_double_after_split = allow_double_after_split
        self.allow_split = allow_split
        self.allow_resplit_aces = allow_resplit_aces
        self.allow_surrender = allow_surrender
        self.hit_soft_17 = hit_soft_17
        self.allow_insurance = allow_insurance

    @staticmethod
    def classic():
        """Standardowe zasady kasynowe."""
        return CasinoRules(
            num_decks=4,
            cut_card_position=0.7,
            blackjack_payout=(3, 2),
            allow_double=True,
            allow_double_after_split=True,
            allow_split=True, 
            allow_resplit_aces=False,
            allow_surrender=True,
            hit_soft_17=True,
            allow_insurance=True
        )

    @staticmethod
    def tight():
        """Kasyno trudniejsze dla gracza."""
        return CasinoRules(
            num_decks=8,
            cut_card_position=0.5,
            blackjack_payout=(6, 5),
            allow_double=False,
            allow_double_after_split=False,
            allow_split=False, 
            allow_resplit_aces=False,
            allow_surrender=False,
            hit_soft_17=True,
            allow_insurance=False
        )

    @staticmethod
    def easy():
        """Kasyno łatwiejsze dla gracza."""
        return CasinoRules(
            num_decks=1,
            cut_card_position=0.95,
            blackjack_payout=(3, 2),
            allow_double=True,
            allow_double_after_split=True,
            allow_split=True, 
            allow_resplit_aces=True,
            allow_surrender=True,
            hit_soft_17=False,
            allow_insurance=True
        )

    @staticmethod
    def vegas_single_deck():
        """Symulacja Vegas Single Deck Blackjack."""
        return CasinoRules(
            num_decks=1,
            cut_card_position=0.95,
            blackjack_payout=(3, 2),
            allow_double=True,
            allow_double_after_split=True,
            allow_split=True, 
            allow_resplit_aces=False,
            allow_surrender=False,
            hit_soft_17=False,
            allow_insurance=True
        )

    def blackjack_multiplier(self):
        """Zwraca współczynnik wypłaty blackjacka jako float, np. 1.5 dla 3:2."""
        return self.blackjack_payout[0] / self.blackjack_payout[1]
