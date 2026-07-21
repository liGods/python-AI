from __future__ import annotations

import unittest
from types import SimpleNamespace

from ok_tasks.card_ai.decision.candidate import CandidateDecision
from ok_tasks.card_ai.search.endgame import search_public_endgame, should_search


def _candidate(action: str, *, terminal: bool = False, block: bool = False, turns: int = 3) -> CandidateDecision:
    return CandidateDecision(action, action, "solo", SimpleNamespace(terminal=terminal, enemy_emergency_block=block, worst_remaining_turns=turns, control_card_cost=0), (turns, action), None, {}, {"total": 0})


class TestEndgameSearch(unittest.TestCase):
    def test_forced_win_prefers_terminal_candidate(self):
        context = SimpleNamespace(hand="789", enemy_counts=(3, 8))
        fallback, win = _candidate("8", turns=2), _candidate("9", terminal=True, turns=0)
        result = search_public_endgame(context, (fallback, win), fallback, 50)
        self.assertEqual("9", result.candidate.effective_action)
        self.assertGreater(result.nodes, 0)

    def test_forced_loss_uses_rule_fallback_when_no_win(self):
        context = SimpleNamespace(hand="789", enemy_counts=(4, 8))
        fallback, worse = _candidate("8", turns=2), _candidate("9", turns=5)
        result = search_public_endgame(context, (fallback, worse), fallback, 50)
        self.assertEqual("8", result.candidate.effective_action)

    def test_unique_block_is_selected(self):
        context = SimpleNamespace(hand="789", enemy_counts=(1, 8))
        fallback, block = _candidate("8", turns=2), _candidate("2", block=True, turns=3)
        result = search_public_endgame(context, (fallback, block), fallback, 50)
        self.assertEqual("2", result.candidate.effective_action)

    def test_non_endgame_does_not_start_search(self):
        context = SimpleNamespace(hand="3456789TJQKA", enemy_counts=(12, 13))
        fallback = _candidate("3", turns=5)
        self.assertFalse(should_search(context, (fallback,), fallback))
        self.assertFalse(search_public_endgame(context, (fallback,), fallback).triggered)


if __name__ == "__main__":
    unittest.main()
