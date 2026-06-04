"""
tests/test_jax_env.py

Tests for the pure-JAX TinyHanabi environment.

Run with:
    cd <project_root>
    python -m pytest tests/test_jax_env.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.jax_env.tiny_hanabi import (
    NUM_MOVES, OBS_SIZE, NUM_COLORS, NUM_RANKS, NUM_AGENTS,
    HAND_SIZE, MAX_INFO_TOKENS, MAX_LIFE_TOKENS, DECK_SIZE,
    reset, step, get_obs, legal_mask, State,
)

KEY = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# 1. Static constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_obs_size(self):
        assert OBS_SIZE == 84

    def test_num_moves(self):
        assert NUM_MOVES == 8

    def test_deck_size(self):
        assert DECK_SIZE == 8  # (3+1)*2


# ---------------------------------------------------------------------------
# 2. reset()
# ---------------------------------------------------------------------------

class TestReset:

    def test_returns_state(self):
        s = reset(KEY)
        assert isinstance(s, State)

    def test_turn_zero(self):
        s = reset(KEY)
        assert s.turn == 0

    def test_score_zero(self):
        s = reset(KEY)
        assert s.score == 0

    def test_not_terminal(self):
        s = reset(KEY)
        assert not s.terminal

    def test_full_info_tokens(self):
        s = reset(KEY)
        assert int(s.info_tokens.sum()) == MAX_INFO_TOKENS

    def test_full_life_tokens(self):
        s = reset(KEY)
        assert int(s.life_tokens.sum()) == MAX_LIFE_TOKENS

    def test_hands_shape(self):
        s = reset(KEY)
        assert s.player_hands.shape == (NUM_AGENTS, HAND_SIZE, NUM_COLORS, NUM_RANKS)

    def test_hands_non_empty(self):
        s = reset(KEY)
        for p in range(NUM_AGENTS):
            for c in range(HAND_SIZE):
                assert s.player_hands[p, c].sum() == 1.0, (
                    f"player {p} card {c} should be exactly one card"
                )

    def test_first_player_starts(self):
        s = reset(KEY)
        assert int(jnp.argmax(s.cur_player_idx)) == 0

    def test_num_cards_dealt(self):
        s = reset(KEY)
        assert s.num_cards_dealt == NUM_AGENTS * HAND_SIZE

    def test_deck_cards_removed_after_deal(self):
        s = reset(KEY)
        # Dealt positions (indices 0..3) should be zero
        assert s.deck[:NUM_AGENTS * HAND_SIZE].sum() == 0.0

    def test_deck_has_remaining_cards(self):
        s = reset(KEY)
        n_remaining = int(jnp.any(jnp.any(s.deck, axis=1), axis=1).sum())
        assert n_remaining == DECK_SIZE - NUM_AGENTS * HAND_SIZE

    def test_zero_fireworks(self):
        s = reset(KEY)
        assert s.fireworks.sum() == 0.0

    def test_different_keys_give_different_states(self):
        s1 = reset(jax.random.PRNGKey(1))
        s2 = reset(jax.random.PRNGKey(2))
        # With overwhelming probability, shuffled decks differ
        assert not jnp.array_equal(s1.deck, s2.deck)

    def test_same_key_gives_same_state(self):
        s1 = reset(KEY)
        s2 = reset(KEY)
        assert jnp.array_equal(s1.deck, s2.deck)


# ---------------------------------------------------------------------------
# 3. legal_mask()
# ---------------------------------------------------------------------------

class TestLegalMask:

    def test_shape(self):
        s = reset(KEY)
        m = legal_mask(s)
        assert m.shape == (NUM_MOVES,)

    def test_at_least_one_legal(self):
        s = reset(KEY)
        m = legal_mask(s)
        assert m.any()

    def test_all_false_at_terminal(self):
        s = reset(KEY)
        # Force terminal via replace
        s2 = s.replace(terminal=True)
        m = legal_mask(s2)
        assert not m.any()

    def test_hints_illegal_with_no_tokens(self):
        s = reset(KEY)
        s2 = s.replace(info_tokens=jnp.zeros(MAX_INFO_TOKENS, dtype=jnp.int32))
        m = legal_mask(s2)
        # Actions 4,5 (hint color) and 6,7 (hint rank) should all be False
        assert not m[4:].any()

    def test_discards_illegal_with_full_tokens(self):
        # Info tokens already full at reset — but let's explicitly check
        # Discard is illegal if info tokens == MAX (already full, can't gain more)
        # Wait — actually the rule is: discard is ALWAYS legal EXCEPT when
        # info_tokens == MAX_INFO_TOKENS (you can't gain a token you already have).
        # At reset, tokens are full → discard should be illegal.
        s = reset(KEY)
        m = legal_mask(s)
        # Actions 0,1 are discard
        assert not m[0] and not m[1]

    def test_play_legal_for_all_card_slots(self):
        s = reset(KEY)
        m = legal_mask(s)
        # Actions 2,3 are PLAY — should be legal (player has both cards)
        assert m[2] and m[3]

    def test_all_legal_after_spending_info_token(self):
        # After one hint (spending a token), discard becomes legal
        s = reset(KEY)
        m0 = legal_mask(s)
        # Find a legal hint and take it
        hint_idx = int(jnp.argmax(m0[4:].astype(jnp.int32))) + 4
        s2, _, _ = step(s, jnp.array(hint_idx))
        # Now player 1's turn; info tokens = MAX-1 → discard legal
        m1 = legal_mask(s2)
        # At least one discard should now be legal
        assert m1[0] or m1[1]


# ---------------------------------------------------------------------------
# 4. step()
# ---------------------------------------------------------------------------

class TestStep:

    def test_returns_tuple(self):
        s = reset(KEY)
        m = legal_mask(s)
        action = jnp.argmax(m.astype(jnp.int32))
        result = step(s, action)
        assert len(result) == 3

    def test_turn_increments(self):
        s = reset(KEY)
        m = legal_mask(s)
        action = jnp.argmax(m.astype(jnp.int32))
        s2, _, _ = step(s, action)
        assert s2.turn == 1

    def test_player_alternates(self):
        s = reset(KEY)
        m = legal_mask(s)
        action = jnp.argmax(m.astype(jnp.int32))
        s2, _, _ = step(s, action)
        # Player 0 acted → now player 1
        assert int(jnp.argmax(s2.cur_player_idx)) == 1

    def test_play_valid_increases_fireworks(self):
        # Find a state where a play action is valid by constructing one
        # Start fresh and try all play actions until one scores
        s = reset(KEY)
        # Try playing all cards until one is a valid rank-0 card
        scored = False
        for _ in range(30):
            m = legal_mask(s)
            # Try play actions first
            for a in [2, 3]:
                if m[a]:
                    aidx = int(jnp.argmax(s.cur_player_idx))
                    card_idx = a - HAND_SIZE
                    card = s.player_hands[aidx, card_idx]
                    color = int(jnp.argmax(card.sum(axis=1)))
                    rank  = int(jnp.argmax(card.sum(axis=0)))
                    fw_before = int(s.fireworks[color].sum())
                    if rank == fw_before:  # valid play
                        s2, reward, _ = step(s, jnp.array(a))
                        assert reward > 0, "valid play should give positive reward"
                        assert s2.fireworks[color].sum() > s.fireworks[color].sum()
                        scored = True
                        break
            if scored:
                break
            # Take any legal action and move on
            action = jnp.argmax(m.astype(jnp.int32))
            s, _, done = step(s, action)
            if done:
                break

    def test_invalid_play_loses_life(self):
        # Find a play action where the card is NOT the next in sequence
        s = reset(KEY)
        for _ in range(30):
            m = legal_mask(s)
            for a in [2, 3]:
                if m[a]:
                    aidx = int(jnp.argmax(s.cur_player_idx))
                    card_idx = a - HAND_SIZE
                    card = s.player_hands[aidx, card_idx]
                    color = int(jnp.argmax(card.sum(axis=1)))
                    rank  = int(jnp.argmax(card.sum(axis=0)))
                    fw_before = int(s.fireworks[color].sum())
                    if rank != fw_before:  # invalid play
                        lives_before = int(s.life_tokens.sum())
                        s2, _, done = step(s, jnp.array(a))
                        lives_after = int(s2.life_tokens.sum())
                        assert lives_after == lives_before - 1
                        # With max_life_tokens=1, this ends the game
                        assert done
                        return
            action = jnp.argmax(m.astype(jnp.int32))
            s, _, done = step(s, action)
            if done:
                break
        pytest.skip("Could not find an invalid play in 30 steps")

    def test_hint_spends_info_token(self):
        s = reset(KEY)
        m = legal_mask(s)
        hint_actions = [a for a in range(4, 8) if m[a]]
        if not hint_actions:
            pytest.skip("No legal hint in initial state")
        tokens_before = int(s.info_tokens.sum())
        s2, _, _ = step(s, jnp.array(hint_actions[0]))
        assert int(s2.info_tokens.sum()) == tokens_before - 1

    def test_discard_restores_info_token(self):
        # First spend a token via hint, then discard
        s = reset(KEY)
        m = legal_mask(s)
        hint_actions = [a for a in range(4, 8) if m[a]]
        if not hint_actions:
            pytest.skip("No legal hint")
        s, _, _ = step(s, jnp.array(hint_actions[0]))
        # Now it's player 1's turn; info tokens = MAX-1 → discard legal
        m2 = legal_mask(s)
        discard_actions = [a for a in [0, 1] if m2[a]]
        if not discard_actions:
            pytest.skip("No legal discard after hint")
        tokens_before = int(s.info_tokens.sum())
        s2, _, _ = step(s, jnp.array(discard_actions[0]))
        assert int(s2.info_tokens.sum()) == tokens_before + 1

    def test_score_non_negative(self):
        s = reset(KEY)
        for _ in range(20):
            m = legal_mask(s)
            action = jnp.argmax(m.astype(jnp.int32))
            s, r, done = step(s, action)
            assert float(r) >= 0 or s.out_of_lives
            if done:
                break

    def test_terminal_state_step_safe(self):
        # Stepping a terminal state shouldn't raise
        s = reset(KEY)
        s2 = s.replace(terminal=True)
        # legal_mask returns all False; step should not crash if called
        s3, _, _ = step(s2, jnp.array(2))
        assert s3.terminal


# ---------------------------------------------------------------------------
# 5. Full episode
# ---------------------------------------------------------------------------

class TestFullEpisode:

    def _run_episode(self, key, policy="first_legal"):
        s = reset(key)
        total_score = 0
        steps = 0
        while not s.terminal and steps < 200:
            m = legal_mask(s)
            if policy == "first_legal":
                action = jnp.argmax(m.astype(jnp.int32))
            else:
                action = jax.random.choice(
                    jax.random.PRNGKey(steps), NUM_MOVES,
                    p=m.astype(jnp.float32) / m.sum()
                )
            s, r, _ = step(s, action)
            total_score = s.score
            steps += 1
        return s, steps

    def test_episode_terminates(self):
        s, steps = self._run_episode(KEY)
        assert s.terminal
        assert steps < 200

    def test_score_in_valid_range(self):
        for seed in range(5):
            s, _ = self._run_episode(jax.random.PRNGKey(seed))
            assert 0 <= int(s.score) <= NUM_COLORS * NUM_RANKS  # 0..4

    def test_multiple_episodes_no_carry_over(self):
        for seed in range(3):
            s = reset(jax.random.PRNGKey(seed))
            assert s.turn == 0 and s.score == 0

    def test_avg_episode_length_reasonable(self):
        # With MAX_LIFE_TOKENS=1, random play bombs frequently → short episodes.
        # Lower bound is 2 (at least one move per player before game ends).
        lengths = []
        for seed in range(20):
            _, n = self._run_episode(jax.random.PRNGKey(seed))
            lengths.append(n)
        avg = np.mean(lengths)
        assert 2 <= avg <= 20, f"unexpected avg episode length {avg:.1f}"


# ---------------------------------------------------------------------------
# 6. get_obs()
# ---------------------------------------------------------------------------

class TestGetObs:

    def test_shape(self):
        s = reset(KEY)
        obs = get_obs(s, s, jnp.array(0), 0)
        assert obs.shape == (OBS_SIZE,)

    def test_dtype_float32(self):
        s = reset(KEY)
        obs = get_obs(s, s, jnp.array(0), 0)
        assert obs.dtype == jnp.float32

    def test_values_bounded(self):
        s = reset(KEY)
        obs = get_obs(s, s, jnp.array(0), 0)
        assert jnp.all(obs >= 0.0) and jnp.all(obs <= 1.0 + 1e-5)

    def test_first_turn_last_action_zeros(self):
        # turn==0 → last_action segment should be all zeros
        s = reset(KEY)
        obs = get_obs(s, s, jnp.array(0), 0)
        from src.jax_env.tiny_hanabi import HANDS_N, BOARD_N, DISCARDS_N, LAST_ACT_N
        start = HANDS_N + BOARD_N + DISCARDS_N
        last_act = obs[start: start + LAST_ACT_N]
        assert jnp.allclose(last_act, 0.0)

    def test_obs_differs_between_players(self):
        # Player 0 and player 1 see different hands
        s = reset(KEY)
        obs0 = get_obs(s, s, jnp.array(0), 0)
        obs1 = get_obs(s, s, jnp.array(0), 1)
        assert not jnp.allclose(obs0, obs1)

    def test_obs_consistent_after_step(self):
        s0 = reset(KEY)
        m  = legal_mask(s0)
        a  = jnp.argmax(m.astype(jnp.int32))
        s1, _, _ = step(s0, a)
        # turn==1 → last_action should be non-trivial
        obs = get_obs(s1, s0, a, 0)
        from src.jax_env.tiny_hanabi import HANDS_N, BOARD_N, DISCARDS_N, LAST_ACT_N
        start = HANDS_N + BOARD_N + DISCARDS_N
        last_act = obs[start: start + LAST_ACT_N]
        assert last_act.any(), "last_action_feats should be non-zero after turn 1"

    def test_obs_no_nan(self):
        s = reset(KEY)
        for seed in range(5):
            s2 = reset(jax.random.PRNGKey(seed))
            obs = get_obs(s2, s2, jnp.array(0), 0)
            assert not jnp.any(jnp.isnan(obs)), "obs contains NaN"

    def test_belief_feats_sum_to_one(self):
        # Each card slot's belief distribution should sum to ≈1
        s = reset(KEY)
        obs = get_obs(s, s, jnp.array(0), 0)
        from src.jax_env.tiny_hanabi import (
            HANDS_N, BOARD_N, DISCARDS_N, LAST_ACT_N,
            NUM_COLORS, NUM_RANKS, HAND_SIZE,
        )
        belief_start = HANDS_N + BOARD_N + DISCARDS_N + LAST_ACT_N
        belief = obs[belief_start:].reshape(NUM_AGENTS, HAND_SIZE, -1)
        # The first NC*NR dims of each card are the normalized belief
        NR_NC = NUM_COLORS * NUM_RANKS
        for agent_idx in range(NUM_AGENTS):
            for card_idx in range(HAND_SIZE):
                b = belief[agent_idx, card_idx, :NR_NC]
                total = float(b.sum())
                assert abs(total - 1.0) < 1e-4, (
                    f"belief for agent {agent_idx} card {card_idx} sums to {total}"
                )


# ---------------------------------------------------------------------------
# 7. JIT and vmap compatibility
# ---------------------------------------------------------------------------

class TestJitVmap:

    def test_reset_jit(self):
        reset_jit = jax.jit(reset)
        s = reset_jit(KEY)
        assert isinstance(s, State)

    def test_step_jit(self):
        step_jit = jax.jit(step)
        s = reset(KEY)
        m = legal_mask(s)
        a = jnp.argmax(m.astype(jnp.int32))
        s2, r, done = step_jit(s, a)
        assert isinstance(s2, State)

    def test_legal_mask_jit(self):
        s = reset(KEY)
        m = jax.jit(legal_mask)(s)
        assert m.shape == (NUM_MOVES,)

    def test_get_obs_jit(self):
        s = reset(KEY)
        obs = jax.jit(get_obs, static_argnums=(3,))(s, s, jnp.array(0), 0)
        assert obs.shape == (OBS_SIZE,)

    def test_vmap_reset(self):
        """vmap reset over multiple keys → batch of independent states."""
        keys = jax.random.split(KEY, 4)
        states = jax.vmap(reset)(keys)
        assert states.player_hands.shape == (4, NUM_AGENTS, HAND_SIZE, NUM_COLORS, NUM_RANKS)

    def test_vmap_step(self):
        """vmap step over a batch of states with a fixed action."""
        keys  = jax.random.split(KEY, 4)
        states = jax.vmap(reset)(keys)
        actions = jnp.array([2, 2, 3, 3])
        new_states, rewards, dones = jax.vmap(step)(states, actions)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_vmap_get_obs(self):
        keys   = jax.random.split(KEY, 4)
        states = jax.vmap(reset)(keys)
        actions = jnp.zeros(4, dtype=jnp.int32)
        obs_fn  = jax.jit(jax.vmap(lambda s: get_obs(s, s, jnp.array(0), 0)))
        obs = obs_fn(states)
        assert obs.shape == (4, OBS_SIZE)

    def test_jit_compile_once(self):
        """Calling step twice should not recompile (same abstract shape)."""
        import time
        step_jit = jax.jit(step)
        s = reset(KEY)
        m = legal_mask(s)
        a = jnp.argmax(m.astype(jnp.int32))
        # Warm up
        s2, _, _ = step_jit(s, a)
        s2.score.block_until_ready()
        t0 = time.perf_counter()
        s3, _, _ = step_jit(s2, a)
        s3.score.block_until_ready()
        elapsed = time.perf_counter() - t0
        # Threshold is generous to accommodate CPU dispatch overhead.
        assert elapsed < 5.0, f"step took {elapsed:.3f}s — possible recompile?"
