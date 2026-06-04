# src/jax_env/tiny_hanabi.py
# Pure-JAX Tiny Hanabi environment.
#
# Observation encoding follows JaxMARL (obs_size=84 for this config).
# Action space has 8 actions — no NOOP, matching the HLE C++ library.
#
# Config: 2 colors, 2 ranks, 2 players, hand_size=2,
#         max_info_tokens=3, max_life_tokens=1,
#         num_cards_of_rank=[3,1] (standard tiny Hanabi deck → deck_size=8).
#
# Actions (int in [0,7]):
#   0,1 — DISCARD card at index 0,1
#   2,3 — PLAY card at index 0,1
#   4,5 — HINT COLOR 0,1 to opponent
#   6,7 — HINT RANK 0,1 to opponent
#
# Public API (all jit/vmap compatible):
#   reset(key)                                → State
#   step(state, action)                       → (State, float, bool)
#   get_obs(new_state, old_state, action, aid) → jnp.ndarray (84,) float32
#   legal_mask(state)                         → jnp.ndarray (8,)  bool
#   score(state)                              → int

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from functools import partial

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

NUM_COLORS      = 2
NUM_RANKS       = 2
NUM_AGENTS      = 2
HAND_SIZE       = 2
MAX_INFO_TOKENS = 3
MAX_LIFE_TOKENS = 1
CARDS_OF_RANK   = np.array([3, 1], dtype=np.int32)   # copies of rank-r per color
DECK_SIZE       = int(CARDS_OF_RANK.sum() * NUM_COLORS)  # 8
NUM_MOVES       = 8   # no NOOP

# Action boundary indices (inclusive), used as static Python constants
_D_LO = 0;               _D_HI = HAND_SIZE - 1                              # 0,1
_P_LO = HAND_SIZE;        _P_HI = 2 * HAND_SIZE - 1                         # 2,3
_C_LO = 2 * HAND_SIZE;    _C_HI = 2 * HAND_SIZE + NUM_COLORS - 1            # 4,5
_R_LO = _C_HI + 1;        _R_HI = NUM_MOVES - 1                             # 6,7

# ---------------------------------------------------------------------------
# Observation segment sizes (verified: sum = 84)
# ---------------------------------------------------------------------------

HANDS_N     = (NUM_AGENTS - 1) * HAND_SIZE * NUM_COLORS * NUM_RANKS + NUM_AGENTS   # 10
BOARD_N     = ((DECK_SIZE - NUM_AGENTS * HAND_SIZE)
               + NUM_COLORS * NUM_RANKS
               + MAX_INFO_TOKENS + MAX_LIFE_TOKENS)                                 # 12
DISCARDS_N  = NUM_COLORS * int(CARDS_OF_RANK.sum())                                 # 8
LAST_ACT_N  = (NUM_AGENTS + 4 + NUM_AGENTS
               + NUM_COLORS + NUM_RANKS
               + HAND_SIZE + HAND_SIZE
               + NUM_COLORS * NUM_RANKS + 1 + 1)                                    # 22
BELIEF_N    = NUM_AGENTS * HAND_SIZE * (NUM_COLORS * NUM_RANKS + NUM_COLORS + NUM_RANKS)  # 32
OBS_SIZE    = HANDS_N + BOARD_N + DISCARDS_N + LAST_ACT_N + BELIEF_N               # 84

assert OBS_SIZE == 84, f"obs_size mismatch: {OBS_SIZE}"

# ---------------------------------------------------------------------------
# Pre-computed static arrays (numpy, treated as compile-time constants)
# ---------------------------------------------------------------------------

# All (color, rank) pairs in the full deck, with repetitions per CARDS_OF_RANK
_FULL_PAIRS: np.ndarray = np.array(
    [[c, r]
     for c in range(NUM_COLORS)
     for r in range(NUM_RANKS)
     for _ in range(CARDS_OF_RANK[r])],
    dtype=np.int32,
)  # (DECK_SIZE=8, 2)

# Total copies of each card type in the full deck
_COUNTS: np.ndarray = np.zeros((NUM_COLORS, NUM_RANKS), dtype=np.int32)
for _c, _r in _FULL_PAIRS:
    _COUNTS[_c, _r] += 1
# _COUNTS = [[3,1],[3,1]]

# ---------------------------------------------------------------------------
# State pytree  (flax.struct.dataclass → frozen, supports .replace())
# ---------------------------------------------------------------------------

@struct.dataclass
class State:
    # Card arrays — one-hot (NUM_COLORS, NUM_RANKS) per slot
    deck:            jnp.ndarray  # (DECK_SIZE, NC, NR)
    discard_pile:    jnp.ndarray  # (DECK_SIZE, NC, NR)
    fireworks:       jnp.ndarray  # (NC, NR)  incremental: [1,1,0] = ranks 0&1 played
    player_hands:    jnp.ndarray  # (NA, HS, NC, NR)
    # Token counts — thermometer: ones at low indices = tokens available
    info_tokens:     jnp.ndarray  # (MAX_INFO_TOKENS,) int {0,1}
    life_tokens:     jnp.ndarray  # (MAX_LIFE_TOKENS,) int {0,1}
    # Knowledge & hints — per agent, per card slot
    card_knowledge:  jnp.ndarray  # (NA, HS, NC*NR)  belief prior [0,1]
    colors_revealed: jnp.ndarray  # (NA, HS, NC)
    ranks_revealed:  jnp.ndarray  # (NA, HS, NR)
    # Turn state
    cur_player_idx:      jnp.ndarray  # (NA,) one-hot
    terminal:            bool
    out_of_lives:        bool
    bombed:              bool
    num_cards_dealt:     int
    num_cards_discarded: int
    last_round_count:    int
    turn:                int
    score:               int


# ---------------------------------------------------------------------------
# Action-type predicates (return JAX scalar bool — safe inside jit/vmap)
# ---------------------------------------------------------------------------

def _is_discard(a):     return (a >= _D_LO) & (a <= _D_HI)
def _is_play(a):        return (a >= _P_LO) & (a <= _P_HI)
def _is_hint_color(a):  return (a >= _C_LO) & (a <= _C_HI)
def _is_hint_rank(a):   return (a >= _R_LO) & (a <= _R_HI)
def _is_hint(a):        return _is_hint_color(a) | _is_hint_rank(a)


def _get_target_and_hint_idx(aidx, action):
    """For a hint action, return (absolute target player, hint color/rank index)."""
    is_color = _is_hint_color(action)
    # 2-player game → only one target offset (1), target = (aidx + 1) % 2
    target = (aidx + 1) % NUM_AGENTS
    hint_idx = jnp.where(is_color, action - _C_LO, action - _R_LO)
    return target.astype(jnp.int32), hint_idx.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Deck helpers
# ---------------------------------------------------------------------------

def _one_hot_encode_deck(pairs: jnp.ndarray) -> jnp.ndarray:
    """(DECK_SIZE, 2) (color,rank) pairs → (DECK_SIZE, NC, NR) one-hot deck."""
    def _enc(_, pair):
        card = jnp.zeros((NUM_COLORS, NUM_RANKS)).at[pair[0], pair[1]].set(1.0)
        return None, card
    _, deck = lax.scan(_enc, None, pairs)
    return deck


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

@jax.jit
def reset(key: jnp.ndarray) -> State:
    """Shuffle a fresh deck and return the initial game state."""
    key, sk = jax.random.split(key)
    shuffled = jax.random.permutation(sk, jnp.array(_FULL_PAIRS), axis=0)
    deck = _one_hot_encode_deck(shuffled)
    return _init_state(deck)


def _init_state(deck: jnp.ndarray) -> State:
    """Build initial State from a pre-shuffled one-hot deck."""
    # Deal HAND_SIZE cards to each player from the top of the deck
    def _deal(_, aidx):
        hand = lax.dynamic_slice(
            deck, (aidx * HAND_SIZE, 0, 0), (HAND_SIZE, NUM_COLORS, NUM_RANKS)
        )
        return None, hand

    _, hands = lax.scan(_deal, None, jnp.arange(NUM_AGENTS))  # (NA, HS, NC, NR)
    n_dealt = NUM_AGENTS * HAND_SIZE  # 4 (Python int — static)

    return State(
        deck            = deck.at[:n_dealt].set(jnp.zeros((NUM_COLORS, NUM_RANKS))),
        discard_pile    = jnp.zeros_like(deck),
        fireworks       = jnp.zeros((NUM_COLORS, NUM_RANKS)),
        player_hands    = hands,
        info_tokens     = jnp.ones(MAX_INFO_TOKENS, dtype=jnp.int32),
        life_tokens     = jnp.ones(MAX_LIFE_TOKENS, dtype=jnp.int32),
        card_knowledge  = jnp.ones((NUM_AGENTS, HAND_SIZE, NUM_COLORS * NUM_RANKS)),
        colors_revealed = jnp.zeros((NUM_AGENTS, HAND_SIZE, NUM_COLORS)),
        ranks_revealed  = jnp.zeros((NUM_AGENTS, HAND_SIZE, NUM_RANKS)),
        cur_player_idx  = jnp.zeros(NUM_AGENTS).at[0].set(1),
        terminal        = False,
        out_of_lives    = False,
        bombed          = False,
        num_cards_dealt     = n_dealt,
        num_cards_discarded = 0,
        last_round_count    = 0,
        turn                = 0,
        score               = 0,
    )


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

@jax.jit
def step(state: State, action: jnp.ndarray):
    """
    Execute `action` for the current player.

    Returns:
        new_state : State
        reward    : float  (incremental score delta; negative if a life is lost)
        done      : bool
    """
    aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
    is_hint = _is_hint(action)

    new_state = lax.cond(
        is_hint,
        lambda s, a: _hint_fn(s, aidx, a),
        lambda s, a: _discard_play_fn(s, aidx, a),
        state, action,
    )

    # ---- Terminal / reward --------------------------------------------------
    fw_before = state.fireworks.sum()
    fw_after  = new_state.fireworks.sum()

    out_of_lives = new_state.life_tokens.sum() == 0
    game_won     = fw_after == NUM_COLORS * NUM_RANKS
    deck_empty   = new_state.num_cards_dealt >= DECK_SIZE
    last_round_count = new_state.last_round_count + deck_empty
    terminal = out_of_lives | game_won | (last_round_count == NUM_AGENTS + 1)

    reward = (~out_of_lives) * (fw_after - fw_before)
    reward = reward - out_of_lives * fw_after * (~state.bombed)
    bombed = out_of_lives | state.bombed

    # Advance player
    next_p = (aidx + 1) % NUM_AGENTS
    cur_player_idx = jnp.zeros(NUM_AGENTS).at[next_p].set(1)

    new_state = new_state.replace(
        terminal         = terminal,
        cur_player_idx   = cur_player_idx,
        out_of_lives     = out_of_lives,
        last_round_count = last_round_count.astype(jnp.int32),
        bombed           = bombed,
        turn             = state.turn + 1,
        score            = state.score + reward.astype(jnp.int32),
    )
    return new_state, reward.astype(jnp.float32), terminal


def _discard_play_fn(state: State, aidx, action) -> State:
    """Execute a DISCARD or PLAY action. Returns updated State (no terminal logic)."""
    is_discard = _is_discard(action)

    hand_before = state.player_hands[aidx]  # (HS, NC, NR)
    card_idx = jnp.where(is_discard, action, action - HAND_SIZE).astype(jnp.int32)
    card = hand_before[card_idx]             # (NC, NR)

    # ---- Info token gain on discard ----------------------------------------
    n_info = state.info_tokens.sum()
    new_n_info = n_info + (is_discard & (n_info < MAX_INFO_TOKENS))
    info_tokens = jnp.where(
        new_n_info > 0,
        state.info_tokens.at[(new_n_info - 1).astype(jnp.int32)].set(1),
        state.info_tokens,
    )

    # ---- Play validity ------------------------------------------------------
    color, rank = jnp.nonzero(card, size=1)
    color = color[0].astype(jnp.int32)
    rank  = rank[0].astype(jnp.int32)
    color_fw = state.fireworks[color]        # (NR,) incremental
    is_valid_play = (~is_discard) & (rank == color_fw.sum().astype(jnp.int32))

    # Gain token for completing a color (playing the highest rank)
    is_final_card = is_valid_play & (rank == NUM_RANKS - 1)
    n_info2   = info_tokens.sum()
    new_n_info2 = n_info2 + (is_final_card & (n_info2 < MAX_INFO_TOKENS))
    info_tokens = jnp.where(
        new_n_info2 > 0,
        info_tokens.at[(new_n_info2 - 1).astype(jnp.int32)].set(1),
        info_tokens,
    )

    # ---- Update fireworks ---------------------------------------------------
    fw_idx = color_fw.sum().astype(jnp.int32)
    new_color_fw = color_fw.at[fw_idx].set(is_valid_play.astype(jnp.float32))
    fireworks = state.fireworks.at[color].set(new_color_fw)

    # ---- Discard the card (always for discard; on invalid play) ------------
    discard_card = is_discard | (~is_valid_play)
    discarded    = jnp.where(discard_card, card, jnp.zeros_like(card))
    discard_pile = state.discard_pile.at[state.num_cards_discarded].set(discarded)
    num_cards_discarded = (state.num_cards_discarded + discard_card).astype(jnp.int32)

    # ---- Life token loss on invalid play ------------------------------------
    life_lost = (~is_discard) & (~is_valid_play)
    n_life    = state.life_tokens.sum().astype(jnp.int32)
    life_tokens = jnp.where(
        life_lost,
        state.life_tokens.at[n_life - 1].set(0),
        state.life_tokens,
    )

    # ---- Remove card from hand; shift remaining left; append new card -------
    # Hint knowledge for removed slot
    p_colors = jnp.delete(state.colors_revealed[aidx], card_idx, axis=0,
                           assume_unique_indices=True)
    p_colors = jnp.append(p_colors, jnp.zeros((1, NUM_COLORS)), axis=0)
    colors_revealed = state.colors_revealed.at[aidx].set(p_colors)

    p_ranks = jnp.delete(state.ranks_revealed[aidx], card_idx, axis=0,
                          assume_unique_indices=True)
    p_ranks = jnp.append(p_ranks, jnp.zeros((1, NUM_RANKS)), axis=0)
    ranks_revealed = state.ranks_revealed.at[aidx].set(p_ranks)

    # Deal a new card (empty array if deck is exhausted or in last round)
    in_last_round = state.last_round_count > 0
    new_card = state.deck[state.num_cards_dealt]  # (NC, NR) — zeros when deck empty
    hand_rest = jnp.delete(hand_before, card_idx, axis=0, assume_unique_indices=True)
    new_hand  = jnp.append(hand_rest, new_card[jnp.newaxis], axis=0)
    player_hands = state.player_hands.at[aidx].set(new_hand)
    deck = state.deck.at[state.num_cards_dealt].set(jnp.zeros((NUM_COLORS, NUM_RANKS)))
    num_cards_dealt = jnp.where(
        in_last_round, state.num_cards_dealt, state.num_cards_dealt + 1
    ).astype(jnp.int32)

    # Knowledge for new card slot
    p_know = jnp.delete(state.card_knowledge[aidx], card_idx, axis=0,
                         assume_unique_indices=True)
    new_know = jnp.where(
        new_card.any(),
        jnp.ones((1, NUM_COLORS * NUM_RANKS)),
        jnp.zeros((1, NUM_COLORS * NUM_RANKS)),
    )
    p_know = jnp.append(p_know, new_know, axis=0)
    card_knowledge = state.card_knowledge.at[aidx].set(p_know)

    return state.replace(
        deck            = deck,
        discard_pile    = discard_pile,
        player_hands    = player_hands,
        card_knowledge  = card_knowledge,
        colors_revealed = colors_revealed,
        ranks_revealed  = ranks_revealed,
        fireworks       = fireworks,
        info_tokens     = info_tokens,
        life_tokens     = life_tokens,
        num_cards_dealt     = num_cards_dealt,
        num_cards_discarded = num_cards_discarded,
    )


def _hint_fn(state: State, aidx, action) -> State:
    """Execute a HINT COLOR or HINT RANK action. Returns updated State."""
    is_color_hint = _is_hint_color(action)
    target, hint_idx = _get_target_and_hint_idx(aidx, action)

    # Build one-hot hint vectors (one is all zeros, the other has a single 1)
    hint_color = (jnp.zeros(NUM_COLORS)
                  .at[jnp.where(is_color_hint, hint_idx, 0)].set(is_color_hint.astype(jnp.float32)))
    hint_rank  = (jnp.zeros(NUM_RANKS)
                  .at[jnp.where(~is_color_hint, hint_idx, 0)].set((~is_color_hint).astype(jnp.float32)))

    cur_knowledge = state.card_knowledge[target]  # (HS, NC*NR)
    cards         = state.player_hands[target]    # (HS, NC, NR)
    card_colors   = cards.sum(axis=2)             # (HS, NC)
    card_ranks    = cards.sum(axis=1)             # (HS, NR)

    # Which cards match the hinted color/rank?
    color_matches = card_colors @ hint_color   # (HS,)
    rank_matches  = card_ranks  @ hint_rank    # (HS,)

    # ---- Negative hints: cards NOT matching → eliminate that color/rank ----
    # Shape logic: outer(HS) × (NC or NR) → (HS, NC/NR); repeat → (HS, NC*NR)
    neg_color = jnp.outer(1 - color_matches, hint_color)          # (HS, NC)
    neg_color = jnp.repeat(neg_color, NUM_COLORS, axis=1).reshape(cur_knowledge.shape)
    neg_rank  = jnp.outer(1 - rank_matches, hint_rank)            # (HS, NR)
    neg_rank  = jnp.repeat(neg_rank, NUM_RANKS, axis=0).reshape(cur_knowledge.shape)

    # ---- Positive hints: cards matching → eliminate other colors/ranks -----
    color_mask = (color_matches * jnp.ones((NUM_COLORS, HAND_SIZE))).T  # (HS, NC)
    pos_color  = color_mask * (1 - hint_color * jnp.ones((HAND_SIZE, NUM_COLORS)))
    pos_color  = jnp.repeat(pos_color, NUM_COLORS, axis=1).reshape(cur_knowledge.shape)

    rank_mask  = (rank_matches * jnp.ones((NUM_RANKS, HAND_SIZE))).T   # (HS, NR)
    pos_rank   = rank_mask * (1 - hint_rank * jnp.ones((HAND_SIZE, NUM_RANKS)))
    pos_rank   = jnp.repeat(pos_rank, NUM_RANKS, axis=0).reshape(cur_knowledge.shape)

    total_color = pos_color + neg_color
    total_rank  = pos_rank  + neg_rank

    new_know_color = (cur_knowledge - is_color_hint.astype(jnp.float32) * total_color).clip(min=0)
    new_know_rank  = (cur_knowledge - (~is_color_hint).astype(jnp.float32) * total_rank).clip(min=0)
    new_knowledge  = jnp.where(is_color_hint, new_know_color, new_know_rank)
    card_knowledge = state.card_knowledge.at[target].set(new_knowledge)

    # Track which cards received the hint (for obs last_action_feats)
    colors_recv = jnp.outer(color_matches, hint_color)
    colors_revealed = state.colors_revealed.at[target].set(
        (state.colors_revealed[target] + colors_recv).clip(max=1)
    )
    ranks_recv  = jnp.outer(rank_matches, hint_rank)
    ranks_revealed = state.ranks_revealed.at[target].set(
        (state.ranks_revealed[target] + ranks_recv).clip(max=1)
    )

    # Consume one info token
    n_info = state.info_tokens.sum().astype(jnp.int32) - 1
    info_tokens = (jnp.arange(MAX_INFO_TOKENS) < n_info).astype(jnp.int32)

    return state.replace(
        card_knowledge  = card_knowledge,
        info_tokens     = info_tokens,
        colors_revealed = colors_revealed,
        ranks_revealed  = ranks_revealed,
    )


# ---------------------------------------------------------------------------
# legal_mask — for the CURRENT player only (no NOOP)
# ---------------------------------------------------------------------------

@jax.jit
def legal_mask(state: State) -> jnp.ndarray:
    """
    Return a (NUM_MOVES,) bool array of legal actions for the current player.
    Illegal actions are False. The result is always all-False for terminal states
    (step should not be called after terminal, but this is safe).
    """
    aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
    my_hand = state.player_hands[aidx]           # (HS, NC, NR)
    has_card = jax.vmap(lambda c: c.any())(my_hand)  # (HS,) bool

    # DISCARD: legal if info tokens not full AND card present
    can_discard = (state.info_tokens.sum() < MAX_INFO_TOKENS) & has_card

    # PLAY: legal if card present
    can_play = has_card

    # HINT COLOR: legal if info tokens available AND opponent has at least one
    #             card of that color
    info_avail = state.info_tokens.sum() > 0
    opp_idx  = (aidx + 1) % NUM_AGENTS
    opp_hand = state.player_hands[opp_idx]          # (HS, NC, NR)
    # Colors present in opponent's hand: (NC,) bool
    opp_colors = opp_hand.sum(axis=(0, 2)).astype(bool)  # sum over HS and NR
    can_hint_color = info_avail & opp_colors            # (NC,) bool

    # HINT RANK: legal if info tokens available AND opponent has at least one
    #            card of that rank
    opp_ranks = opp_hand.sum(axis=(0, 1)).astype(bool)  # sum over HS and NC
    can_hint_rank = info_avail & opp_ranks              # (NR,) bool

    legal = jnp.concatenate([
        can_discard,        # (HS,) = (2,)
        can_play,           # (HS,) = (2,)
        can_hint_color,     # (NC,) = (2,)
        can_hint_rank,      # (NR,) = (2,)
    ])                      # (8,)

    return legal & ~state.terminal


# ---------------------------------------------------------------------------
# get_obs — full 84-dim canonical observation
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(3,))
def get_obs(
    new_state: State,
    old_state: State,
    action: jnp.ndarray,
    aidx: int,
) -> jnp.ndarray:
    """
    Compute the 84-dim canonical observation for player `aidx`.

    Follows JaxMARL's encoding:
      [hands(10) | board(12) | discards(8) | last_action(22) | v0_belief(32)]

    Args:
        new_state : state AFTER the action was taken
        old_state : state BEFORE the action (needed for last_action features)
        action    : the action that produced new_state from old_state.
                    Pass any value (e.g. 0) if new_state.turn == 0.
        aidx      : observing player index (Python int — static)
    """
    hands_feats       = _get_hands_feats(new_state, aidx)
    board_feats       = _get_board_feats(new_state)
    discard_feats     = _binarize_discard_pile(new_state.discard_pile)
    last_action_feats = jnp.where(
        new_state.turn == 0,
        jnp.zeros(LAST_ACT_N),
        _get_last_action_feats(new_state, old_state, action, aidx),
    )
    belief_feats      = _get_belief_feats(new_state, aidx)

    return jnp.concatenate([
        hands_feats,
        board_feats,
        discard_feats,
        last_action_feats,
        belief_feats,
    ]).astype(jnp.float32)


def _get_hands_feats(state: State, aidx: int) -> jnp.ndarray:
    """
    10-dim: opponent hand cards (one-hot, flattened) + missing-card indicators.

    Layout: [other_agents_hands_flat(8), missing_cards_per_agent(2)]
    """
    # Roll so that the observing player is index 0, then drop their hand (unknown)
    rolled = jnp.roll(state.player_hands, -aidx, axis=0)  # (NA, HS, NC, NR)
    other_hands = jnp.delete(rolled, 0, axis=0, assume_unique_indices=True)  # (NA-1, HS, NC, NR)
    other_hands_flat = other_hands.ravel()  # ((NA-1)*HS*NC*NR,) = (8,)

    # missing_cards[i] = True if any card slot for rolled agent i is empty
    has_card_per_slot = jnp.any(jnp.any(rolled, axis=-1), axis=-1)  # (NA, HS) bool
    missing = ~jnp.all(has_card_per_slot, axis=-1)                  # (NA,) bool
    return jnp.concatenate([other_hands_flat, missing.astype(jnp.float32)])


def _get_board_feats(state: State) -> jnp.ndarray:
    """
    12-dim: deck thermometer(4) + fireworks one-hot(4) + info_tokens(3) + life_tokens(1).
    """
    # Deck thermometer: how many cards remain? (4 bits for DECK_SIZE - NA*HS = 4 slots)
    deck_present = jnp.any(jnp.any(state.deck, axis=1), axis=1).astype(jnp.float32)  # (8,)
    # Take the last DECK_SIZE - NA*HS entries in reverse (matches JaxMARL's slice)
    deck_therm = deck_present[DECK_SIZE - 1 : NUM_AGENTS * HAND_SIZE - 1 : -1]  # (4,)

    # Fireworks: convert incremental to one-hot of current level
    def _keep_last_one(x):
        # x: (NR,) incremental vector like [1,1,0]
        # Returns one-hot of the highest 1, e.g. [0,1,0]
        last_one_pos = x.size - 1 - jnp.argmax(jnp.flip(x))
        return jnp.where(jnp.arange(x.size) < last_one_pos, 0.0, x)

    fw_oh = jax.vmap(_keep_last_one)(state.fireworks)  # (NC, NR)

    return jnp.concatenate([
        deck_therm,
        fw_oh.ravel(),
        state.info_tokens.astype(jnp.float32),
        state.life_tokens.astype(jnp.float32),
    ])


def _binarize_discard_pile(discard_pile: jnp.ndarray) -> jnp.ndarray:
    """
    8-dim: thermometer-encoded discard counts per (color, rank).

    For each color: [rank-0 thermometer (3 bits), rank-1 thermometer (1 bit)] = 4 bits.
    Total: 2 colors × 4 bits = 8 bits.
    """
    counts = discard_pile.sum(axis=0)  # (NC, NR) — how many of each card discarded

    def _color_bits(color_counts):
        # color_counts: (NR=2,) — [count_rank0, count_rank1]
        bits_r0 = (jnp.arange(CARDS_OF_RANK[0]) < color_counts[0]).astype(jnp.float32)
        bits_r1 = (jnp.arange(CARDS_OF_RANK[1]) < color_counts[1]).astype(jnp.float32)
        return jnp.concatenate([bits_r0, bits_r1])  # (4,)

    return jax.vmap(_color_bits)(counts).ravel()  # (NC * 4 = 8,)


def _get_last_action_feats(
    new_state: State,
    old_state: State,
    action: jnp.ndarray,
    aidx: int,
) -> jnp.ndarray:
    """
    22-dim encoding of the last action taken (the transition old→new).

    Layout (matches JaxMARL get_last_action_feats):
      acting_player_relative_idx (2)
      move_type                  (4) [play, discard, hint_color, hint_rank]
      target_player_relative_idx (2)
      color_revealed             (2)
      rank_revealed              (2)
      reveal_outcome             (2) which cards were affected
      pos_played_discarded       (2) which hand slot was played/discarded
      played_discarded_card      (4) one-hot card identity
      card_played_score          (1) bool: did the play increase fireworks?
      added_info_token           (1) bool: did playing complete a color?
    """
    acting_abs = jnp.nonzero(old_state.cur_player_idx, size=1)[0][0]
    # Relative acting player index (observer = 0)
    acting_rel = jnp.roll(old_state.cur_player_idx, -aidx)  # (NA,)

    target, hint_idx = _get_target_and_hint_idx(acting_abs, action)
    target_oh  = jnp.zeros(NUM_AGENTS).at[target].set(1.0)
    target_rel = jnp.roll(target_oh, -aidx)  # (NA,)
    target_rel = jnp.where(_is_hint(action), target_rel, jnp.zeros(NUM_AGENTS))

    # Move type one-hot: [play, discard, hint_color, hint_rank]
    move_type = jnp.array([
        _is_play(action),
        _is_discard(action),
        _is_hint_color(action),
        _is_hint_rank(action),
    ], dtype=jnp.float32)

    # Color / rank revealed
    color_revealed = jnp.where(
        _is_hint_color(action),
        jnp.zeros(NUM_COLORS).at[hint_idx].set(1.0),
        jnp.zeros(NUM_COLORS),
    )
    rank_revealed = jnp.where(
        _is_hint_rank(action),
        jnp.zeros(NUM_RANKS).at[hint_idx].set(1.0),
        jnp.zeros(NUM_RANKS),
    )

    # Which cards in target's hand match the hint?
    target_hand = new_state.player_hands[target]  # (HS, NC, NR)
    card_colors = target_hand.sum(axis=2)          # (HS, NC)
    card_ranks  = target_hand.sum(axis=1)          # (HS, NR)
    color_match = (card_colors @ color_revealed).astype(bool)  # (HS,)
    rank_match  = (card_ranks  @ rank_revealed).astype(bool)   # (HS,)
    reveal_outcome = (color_match | rank_match).astype(jnp.float32)  # (HS=2,)

    # Which slot was played / discarded?
    pd_idx = jnp.where(_is_discard(action), action, action - HAND_SIZE)
    pos_pd = (jnp.arange(HAND_SIZE) == pd_idx).astype(jnp.float32)  # (HS=2,)
    pos_pd = jnp.where(_is_hint(action), jnp.zeros(HAND_SIZE), pos_pd)

    # Card identity of what was played / discarded
    actor_hand_before = old_state.player_hands[acting_abs]   # (HS, NC, NR)
    pd_card_slot = jnp.nonzero(pos_pd, size=1)[0][0]
    played_card  = jnp.where(
        pos_pd.any(),
        actor_hand_before[pd_card_slot].ravel(),
        jnp.zeros(NUM_COLORS * NUM_RANKS),
    )

    # Did the play score? Did it add a token?
    scored = (new_state.fireworks.sum() != old_state.fireworks.sum()).astype(jnp.float32)
    added_token = jnp.where(
        _is_play(action),
        (new_state.info_tokens.sum() > old_state.info_tokens.sum()).astype(jnp.float32),
        0.0,
    )

    return jnp.concatenate([
        acting_rel.astype(jnp.float32),
        move_type,
        target_rel.astype(jnp.float32),
        color_revealed,
        rank_revealed,
        reveal_outcome,
        pos_pd,
        played_card.astype(jnp.float32),
        jnp.array([scored]),
        jnp.array([added_token]),
    ])


def _get_belief_feats(state: State, aidx: int) -> jnp.ndarray:
    """
    32-dim v0 belief features.

    For each of the NA × HS card slots (relative to observer):
      normalised_knowledge (NC*NR=4) + color_hint (NC=2) + rank_hint (NR=2) = 8
    Total: 2 × 2 × 8 = 32.

    Normalisation: knowledge_prior × remaining_card_counts,
                   then row-normalised per card slot.
    """
    counts_jax = jnp.array(_COUNTS)  # (NC, NR) — total copies in full deck
    remaining = (
        counts_jax.ravel()
        - state.discard_pile.sum(axis=0).ravel()
        - state.fireworks.ravel()
    )  # (NC*NR=4,) — how many of each card type still in play

    def _belief_per_hand(knowledge, color_hint, rank_hint):
        # knowledge:   (HS, NC*NR)
        # color_hint:  (HS, NC)
        # rank_hint:   (HS, NR)
        weighted = knowledge * remaining                # (HS, NC*NR)
        denom    = weighted.sum(axis=1, keepdims=True)  # (HS, 1)
        normed   = weighted / jnp.where(denom > 0, denom, 1.0)
        normed   = jnp.where(knowledge.any(axis=1, keepdims=True), normed, 0.0)
        return jnp.concatenate([normed, color_hint, rank_hint], axis=-1).ravel()

    # Roll all per-agent arrays so the observer sees themselves at index 0
    roll = lambda x: jnp.roll(x, -aidx, axis=0)
    belief = jax.vmap(_belief_per_hand)(
        roll(state.card_knowledge),   # (NA, HS, NC*NR)
        roll(state.colors_revealed),  # (NA, HS, NC)
        roll(state.ranks_revealed),   # (NA, HS, NR)
    )  # (NA, HS*(NC*NR+NC+NR)) = (2, 16)

    return belief.ravel()  # (32,)


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def score(state: State) -> int:
    """Return the current team score (0–4 for tiny Hanabi)."""
    return state.score
