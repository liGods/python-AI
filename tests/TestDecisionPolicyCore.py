import unittest

from ok_tasks.RlCardRuleModel import load_model, predict
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.policies import StableRulePolicy
from ok_tasks.card_ai.schema import POSITIONS


class TestDecisionPolicyCoreAdapters(unittest.TestCase):
    """The simulator must make the same card choice as the live public adapter."""

    def _live_state(self, engine: BaiJiangPaiEngine, actor: str) -> dict:
        observation = engine.observe(actor)
        positions = tuple(seat for seat in POSITIONS if seat != actor)
        counts = dict(zip(positions, observation.opponent_card_counts))
        enemies = positions if actor == observation.landlord else (observation.landlord,)
        allies = tuple(seat for seat in positions if seat not in enemies)
        state = observation.to_legacy_model_state()
        state.update(
            {
                "position": actor,
                "landlord": observation.landlord,
                "enemy_card_counts": [counts[seat] for seat in enemies],
                "teammate_card_count": counts.get(allies[0]) if allies else None,
                "table_is_teammate": observation.trick_owner in allies,
                "table_is_enemy": observation.trick_owner in enemies,
                "history": [list(event.get("ranks", ())) for event in observation.history if event.get("kind") == "play"],
            }
        )
        return state

    def _assert_same_public_choice(self, engine: BaiJiangPaiEngine) -> None:
        actor = engine.state.current_player
        actions = engine.legal_actions(actor)
        selected = StableRulePolicy().select(engine, actions)
        live_cards = predict(load_model(None), self._live_state(engine, actor))
        self.assertEqual(list(selected.ranks), live_cards, msg=f"actor={actor}, action={selected.to_dict()}")

    def test_landlord_leads_with_same_public_state(self):
        for seed in range(10):
            self._assert_same_public_choice(BaiJiangPaiEngine.create(seed))

    def test_following_turn_uses_same_public_state(self):
        for seed in range(10, 20):
            engine = BaiJiangPaiEngine.create(seed)
            lead = StableRulePolicy().select(engine, engine.legal_actions())
            engine.step(lead)
            self._assert_same_public_choice(engine)

