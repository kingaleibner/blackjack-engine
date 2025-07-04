from models.shoe import Shoe
from game.dealer import Dealer
from game.casino_rules import CasinoRules 
from sim.sim_engine import SimulationResult
from sim.round_summary import RoundSummary, HandResult
import logging

class Table:
    """Zarządza pełną rozgrywką przy stole blackjacka."""

    def __init__(self, rules: CasinoRules = CasinoRules.classic(), players=None, min_bet=10, 
                 strategy_func=None, result: SimulationResult = None):
        self.rules = rules
        self.deck = Shoe(rules)
        self.dealer = Dealer(rules)
        self.players = players if players else []
        self.min_bet = min_bet
        self.round_number = 0
        self.strategy_func = strategy_func
        self.simulation_ended = False
        self.result: SimulationResult = result
        self._logged_hands = set()
        self.expert_logger = None

    def add_player(self, player):
        self.players.append(player)

    def has_active_players(self):
        return any(player.can_continue(self.min_bet) for player in self.players)

    def handle_result(self, player, i, hand, round_summary: RoundSummary):
        key = (player.name, i, self.round_number)
        if key in self._logged_hands:
            return
        self._logged_hands.add(key)

        if i >= len(player.active_bets):
            return

        bet = player.active_bets[i]
        dealer_value = self.dealer.hand.value()
        player_value = hand.value()
        is_blackjack = hand.is_blackjack()

        if is_blackjack and self.dealer.hand.is_blackjack():
            outcome = "pushes"
            net = 0
            player.push(i)
        elif is_blackjack:
            outcome = "wins"
            net = int(bet * self.rules.blackjack_multiplier())
            player.win(i, blackjack=True)
        elif hand.is_busted():
            outcome = "losses"
            net = -bet
            player.lose(i)
        elif self.dealer.hand.is_busted() or player_value > dealer_value:
            outcome = "wins"
            net = bet
            player.win(i)
        elif player_value < dealer_value:
            outcome = "losses"
            net = -bet
            player.lose(i)
        else:
            outcome = "pushes"
            net = 0
            player.push(i)

        if self.result:
            self.result.record_hand(
                player_name=player.name,
                outcome=outcome,
                blackjack=is_blackjack if outcome == "wins" else False
            )

        if self.expert_logger:
            self.expert_logger.update_entry(
                player_name=player.name,
                round_num=self.round_number,
                hand_value=player_value,
                outcome=outcome,
                reward=net
            )
        if hasattr(self.result, "update_expert_decision"):
            self.result.update_expert_decision(
                player_name=player.name,
                round_num=self.round_number,
                hand_index=i,
                outcome=outcome,
                reward=net
            )

        round_summary.add_result(
            HandResult(
                player_name=player.name,
                hand_index=i,
                outcome=outcome,
                reward=net,
                blackjack=is_blackjack,
                hand_value=player_value,
                dealer_value=dealer_value
            )
        )

    def play_round(self) -> RoundSummary:
        self.round_number += 1
        round_summary = RoundSummary(round_number=self.round_number)
        # print(f"\n========== Runda {self.round_number} ==========\n")

        self.dealer.new_round()
        for player in self.players:
            player.new_round()

        active_players = []
        bets = {}

        # Pierwsze karty
        for player in self.players:
            player.hands[0].add_card(self.deck.draw())
        self.dealer.hand.add_card(self.deck.draw())

        # Zakłady
        for player in self.players:
            if player.can_continue(self.min_bet):
                try:
                    bet_amount = player.strategy.bet_size(self.min_bet) if hasattr(player.strategy, "bet_size") else self.min_bet
                    player.place_bet(bet_amount)
                    bets[player] = bet_amount
                    round_summary.bets[player.name] = bet_amount
                    active_players.append(player)
                except ValueError:
                    if self.expert_logger:
                        self.expert_logger.warning(f"{player.name} nie mógł postawić zakładu.")
            else:
                if self.expert_logger:
                    self.expert_logger.info(f"{player.name} opuszcza grę (brak środków).")

        if not active_players:
            return round_summary

        # Następne karty
        for player in active_players:
            player.hands[0].add_card(self.deck.draw())
        self.dealer.hand.add_card(self.deck.draw())

        tc_snapshot = self.deck.true_count()
        round_summary.true_count = self.deck.true_count()

        visible = [c for p in active_players for c in p.hands[0].cards] + [self.dealer.hand.cards[0]]
        if hasattr(self.deck, "observe_cards"):
            self.deck.observe_cards(visible)

        # Ubezpieczenie
        if self.dealer.hand.cards[0].rank == 'A' and self.rules.allow_insurance:
            for player in active_players:
                for i, hand in enumerate(player.hands):
                    if player.strategy.want_insurance(hand, self.dealer.hand.cards[0]):
                        player.take_insurance(i)

            if self.dealer.hand.is_blackjack():
                for player in active_players:
                    for i, hand in enumerate(player.hands):
                        if i in player.insurance_bets:
                            payout = player.insurance_bets[i] * 2
                            player.chips.payout(payout)
                        self.handle_result(player, i, hand, round_summary)
                return round_summary

        # Ruchy graczy
        for player in active_players:
            i = 0
            while i < len(player.hands):
                hand = player.hands[i]
                dealer_card = self.dealer.upcard()
                seq = 1
                while not hand.is_busted():
                    hand.has_moved = True
                    move = player.strategy.decide(hand, dealer_card)
                    hand.final_action = move

                    if self.expert_logger:
                        self.expert_logger.log(hand, dealer_card, self.deck.true_count(), move, 0, None, seq)

                    tc_now = self.deck.true_count()
                    self.result.record_expert_decision(
                        player_name=player.name,
                        round_num=self.round_number,
                        hand_index=i,
                        true_count=tc_now,
                        hand=hand,
                        dealer_card=dealer_card,
                        action=move,
                        bet_amount=bets[player],
                        decks_left=self.deck.decks_left()
                    )

                    seq += 1

                    if move == "hit":
                        hand.add_card(self.deck.draw())
                        if hand.is_busted():
                            break
                    elif move == "stand":
                        break
                    elif move == "double" and self.rules.allow_double:
                        current_bet = player.active_bets[i]
                        if player.chips.total() >= current_bet:
                            player.place_bet(current_bet)
                            round_summary.bets[player.name] += current_bet
                            player.active_bets[i] *= 2
                            hand.add_card(self.deck.draw())
                            hand.is_active = False
                        else:
                            hand.add_card(self.deck.draw())
                        break
                    elif move == "split" and self.rules.allow_split:
                        current_bet = player.active_bets[i]
                        if player.chips.total() >= current_bet:
                            player.split_hand(i, self.deck)
                        else:
                            hand.add_card(self.deck.draw())
                        break
                    elif move == "surrender" and self.rules.allow_surrender and hand.can_surrender():
                        player.surrender_hand(i)
                        break
                i += 1

        # Ruchy krupiera
        self.dealer.play(self.deck)

        # Wyniki
        for player in active_players:
            for i, hand in enumerate(player.hands):
                if not hand.is_busted():
                    self.handle_result(player, i, hand, round_summary)

        # Reshuffle
        if self.deck.needs_reshuffle():
            self.deck.reshuffle()

        self.simulation_ended = not self.has_active_players()
        self._logged_hands.clear()
        if self.expert_logger:
            self.expert_logger.flush()

        return round_summary
