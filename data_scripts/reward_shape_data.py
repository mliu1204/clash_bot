"""
Build trajectory CSV from proper_meta_tight.csv + paired_replay_data.csv.
Output columns: battle_id, x, y, card, time, side, card_index, level, ability,
card_type, player_id, hand_0..3, deck_0..7, reward.
Reward = ±1 / (total actions by that side in the battle); + if that side won, - otherwise.
"""

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META_CSV = ROOT / "data/check_point_data/3_3_2_donthave225/proper_meta_tight.csv"
REPLAY_CSV = ROOT / "data/check_point_data/3_3_2_donthave225/paired_replay_data.csv"
SAVE_CSV = ROOT / "data/ready_data/traj_win.csv"

EXCLUDED_CARDS = frozenset({
    "warmth", "party-rocket", "super-knight", "wizard-trio", "super-mini-pekka",
    "super-witch", "rocket-silo", "super-archers", "party-hut", "super-magic-archer",
    "barbarian-launcher",
})


def _norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower().replace(" ", "-")


def _deck_and_map(meta_row, prefix):
    """List of 8 display names and norm->display map for one team."""
    deck, m = [], {}
    for i in range(8):
        name = meta_row.get(f"{prefix}_cards_{i}_name")
        if pd.isna(name) or not str(name).strip():
            deck.append("")
            continue
        norm = _norm(name)
        evo = meta_row.get(f"{prefix}_cards_{i}_evolutionLevel")
        has_evo = pd.notna(evo) and str(evo).strip() != ""
        if i in (0, 1) and has_evo:
            display = f"evo-{norm}"
        elif i in (2, 3) and has_evo:
            display = f"hero-{norm}"
        else:
            display = norm
        deck.append(display)
        if norm:
            m[norm] = display
    return deck, m


def _has_excluded(actions):
    norms = {_norm(c) for c in actions["card"].dropna().astype(str)}
    return bool(norms & EXCLUDED_CARDS)


def _fill_hand(actions, is_team, deck_list, place_col):
    """Fill hand_0..3 using sliding window over place sequence (non-ability only)."""
    subset = actions.loc[is_team]
    seq = [c for c in subset["card"] if "ability" not in str(c)]
    nonempty = [c for c in deck_list if c]
    start = random.sample(nonempty, min(4, len(nonempty))) if nonempty else []
    while len(start) < 4:
        start.append("")
    seq = start + seq
    idx = 0
    for row in subset.itertuples():
        if "ability" not in str(getattr(row, "card", "")):
            idx += 1
        in_hand = seq[idx : idx + 4]
        hand = [c for c in deck_list if c and c not in in_hand]
        while len(hand) < 4:
            hand.append("")
        for i in range(4):
            actions.loc[row.Index, f"hand_{i}"] = hand[i]


def build_traj(meta_path=META_CSV, replay_path=REPLAY_CSV, save_path=SAVE_CSV, max_battles=None):
    meta = pd.read_csv(str(meta_path))
    meta = meta[meta["deckSelection"] == "collection"]
    meta = meta[meta["team_0_trophyChange"].notna() | meta["opponent_0_trophyChange"].notna()]
    if max_battles:
        meta = meta.head(max_battles)

    replay = pd.read_csv(str(replay_path))
    replay = replay[replay["card"].astype(str).str.strip() != "_invalid"]
    replay = replay[replay["x"].notna() & replay["y"].notna()]
    replay_by_battle = replay.groupby("battle_id", sort=False)

    out_cols = (
        ["battle_id", "x", "y", "card", "time", "side", "card_index", "level", "ability", "card_type", "player_id"]
        + [f"hand_{i}" for i in range(4)] + [f"deck_{i}" for i in range(8)] + ["reward"]
    )
    chunks = []

    for battle in meta.itertuples(index=False):
        bid = str(getattr(battle, "replayTag")).strip().lstrip("#")
        try:
            actions = replay_by_battle.get_group(bid).copy()
        except KeyError:
            continue
        if actions.empty or _has_excluded(actions):
            continue

        b = battle._asdict()
        t0_deck, t0_map = _deck_and_map(b, "team_0")
        o0_deck, o0_map = _deck_and_map(b, "opponent_0")

        is_t = actions["side"].astype(str).str.strip() == "t"
        norm_cards = actions["card"].astype(str).str.strip().str.lower().str.replace(" ", "-", regex=False)
        actions.loc[is_t, "card"] = norm_cards.loc[is_t].map(t0_map).fillna(actions.loc[is_t, "card"])
        actions.loc[~is_t, "card"] = norm_cards.loc[~is_t].map(o0_map).fillna(actions.loc[~is_t, "card"])

        # card_index, level, ability, card_type
        def slot_and_level(row):
            side_deck = t0_deck if str(row["side"]).strip() == "t" else o0_deck
            prefix = "team_0" if str(row["side"]).strip() == "t" else "opponent_0"
            c = str(row["card"])
            idx = next((i for i, d in enumerate(side_deck) if d == c), 0)
            lvl = b.get(f"{prefix}_cards_{idx}_level")
            lvl = lvl if pd.notna(lvl) else ""
            return idx, lvl

        actions["ability"] = actions["card"].astype(str).str.lower().str.contains("ability", na=False).astype(int)
        actions["card_type"] = ""
        idx_level = actions.apply(slot_and_level, axis=1)
        actions["card_index"] = [x[0] for x in idx_level]
        actions["level"] = [x[1] for x in idx_level]

        _fill_hand(actions, is_t, t0_deck, "card")
        _fill_hand(actions, ~is_t, o0_deck, "card")
        for i in range(8):
            actions[f"deck_{i}"] = o0_deck[i]
            actions.loc[is_t, f"deck_{i}"] = t0_deck[i]

        n_t = max(1, is_t.sum())
        n_o = max(1, (~is_t).sum())
        t0_won = (getattr(battle, "team_0_crowns", 0) or 0) > (getattr(battle, "opponent_0_crowns", 0) or 0)
        actions["reward"] = (1.0 if t0_won else -1.0) / n_t
        actions.loc[~is_t, "reward"] = (1.0 if not t0_won else -1.0) / n_o

        chunks.append(actions[out_cols])

    pd.concat(chunks, ignore_index=True).to_csv(str(save_path), index=False)
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    build_traj(save_path=SAVE_CSV)
