

import csv
from datetime import datetime
import random
import pandas as pd
from collections import deque


meta_csv = "/Users/michaelliu/Documents/clash_bot/data/check_point_data/3_3_2_donthave225/proper_meta_tight.csv"
replay_csv = "/Users/michaelliu/Documents/clash_bot/data/check_point_data/3_3_2_donthave225/paired_replay_data.csv"
save_csv = "/Users/michaelliu/Documents/clash_bot/data/ready_data/traj.csv"
okezue_placements = "/Users/michaelliu/Documents/clash_bot/data/okezue_data/all_card_placements_part1.txt"
okezue_placements_part2 = "/Users/michaelliu/Documents/clash_bot/data/okezue_data/all_card_placements_part2.txt"
card_counts_csv = "/Users/michaelliu/Documents/clash_bot/data/okezue_data/card_counts.csv"

# Cards that disqualify a battle if any appear in it (from card_counts.csv tail / custom list)
EXCLUDED_CARDS = frozenset({
    "warmth", "party-rocket", "super-knight", "wizard-trio", "super-mini-pekka",
    "super-witch", "rocket-silo", "super-archers", "party-hut", "super-magic-archer",
    "barbarian-launcher",
})


def _normalize_card_name(name):
    """Lowercase, strip, replace spaces with hyphens for matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return str(name).strip().lower().replace(" ", "-")


def _card_display_name(battle: dict, team_prefix: str, slot: int) -> str | None:
    """Return display name (evo-X / hero-X / X) for deck slot, or None if no name."""
    name = battle.get(f"{team_prefix}_cards_{slot}_name")
    if pd.isna(name) or name is None or str(name).strip() == "":
        return None
    norm = _normalize_card_name(name)
    if not norm:
        return None
    evo_level = battle.get(f"{team_prefix}_cards_{slot}_evolutionLevel")
    if slot in (0, 1) and pd.notna(evo_level) and str(evo_level).strip() != "":
        return f"evo-{norm}"
    if slot in (2, 3) and pd.notna(evo_level) and str(evo_level).strip() != "":
        return f"hero-{norm}"
    return norm


def _build_team_card_map(battle, team_prefix: str) -> dict[str, str]:
    """Map normalized card name -> display name (evo-/hero- prefixed) for one team."""
    out = {}
    for slot in range(8):
        display = _card_display_name(battle, team_prefix, slot)
        if display:
            norm = _normalize_card_name(battle.get(f"{team_prefix}_cards_{slot}_name"))
            if norm:
                out[norm] = display
    return out


def _build_team_deck_list(battle: dict, team_prefix: str) -> list[str]:
    """Ordered list of 8 card display names (evo-/hero- prefixed) for one team."""
    return [
        _card_display_name(battle, team_prefix, slot) or ""
        for slot in range(8)
    ]


def _normalize_card_series(s: pd.Series) -> pd.Series:
    """Vectorized normalize for a column of card names."""
    return s.astype(str).str.strip().str.lower().str.replace(" ", "-", regex=False)


def _battle_has_excluded_cards(actions: pd.DataFrame, excluded_cards: frozenset | set | None) -> bool:
    """True if any card in actions (normalized) is in excluded_cards."""
    if not excluded_cards:
        return False
    norms = set(_normalize_card_name(c) for c in actions["card"].dropna().astype(str))
    return bool(norms & set(excluded_cards))


def _infer_deck_from_plays(card_series: pd.Series, skip_ability: bool = True) -> list[str]:
    """Infer deck (up to 8 cards) from order of first appearance in plays. No meta needed."""
    seen = []
    seen_set = set()
    for c in card_series:
        name = _normalize_card_name(c)
        if not name or name == "nan" or (skip_ability and "ability" in str(c).lower()):
            continue
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)
            if len(seen) >= 8:
                break
    while len(seen) < 8:
        seen.append("")
    return seen[:8]


def build_traj_csv(
    meta_path: str = meta_csv,
    replay_path: str = replay_csv,
    save_path: str = save_csv,
    max_battles: int | None = None,
    excluded_cards: frozenset | set | None = EXCLUDED_CARDS,
):
    meta = pd.read_csv(meta_path)
    replay = pd.read_csv(replay_path)
    # Only rows with deckSelection = collection and at least one trophy change present
    meta = meta[meta["deckSelection"] == "collection"]
    has_t0 = meta["team_0_trophyChange"].notna()
    has_o0 = meta["opponent_0_trophyChange"].notna()
    meta = meta[has_t0 | has_o0]

    # Optionally limit to a smaller number of battles for faster testing
    if max_battles is not None:
        meta = meta.head(max_battles)

    # Group replay by battle_id once (avoids scanning full replay per battle)
    replay_by_battle = replay.groupby("battle_id", sort=False)

    traj_chunks = []
    for battle in meta.itertuples(index=False):
        battle_id = str(getattr(battle, "replayTag")).strip().lstrip("#")
        try:
            actions = replay_by_battle.get_group(battle_id).copy()
        except KeyError:
            continue
        if actions.empty:
            continue
        if _battle_has_excluded_cards(actions, excluded_cards):
            continue
        battle_dict = battle._asdict()
        team_0_map = _build_team_card_map(battle_dict, "team_0")
        opponent_0_map = _build_team_card_map(battle_dict, "opponent_0")
        team_0_deck_list = _build_team_deck_list(battle_dict, "team_0")
        opponent_0_deck_list = _build_team_deck_list(battle_dict, "opponent_0")
        # Vectorized: normalize cards and map by team
        norm_cards = _normalize_card_series(actions["card"])
        is_team_0 = actions["side"].astype(str).str.strip() == "t"
        actions = actions.copy()
        actions.loc[is_team_0, "card"] = norm_cards.loc[is_team_0].map(team_0_map).fillna(actions.loc[is_team_0, "card"])
        actions.loc[~is_team_0, "card"] = norm_cards.loc[~is_team_0].map(opponent_0_map).fillna(actions.loc[~is_team_0, "card"])

        # Approximate hand state for team_0 over time based on played cards.
        team_actions = actions.loc[is_team_0]
        place_sequence = list(team_actions["card"])
        place_sequence = [action for action in place_sequence if "ability" not in str(action)]
        # Prepend 4 random cards as initial hand, then sliding window over [start_4, ...plays]
        team_nonempty = [c for c in team_0_deck_list if c]
        start_4 = random.sample(team_nonempty, min(4, len(team_nonempty))) if team_nonempty else []
        while len(start_4) < 4:
            start_4.append("")
        place_sequence = start_4 + place_sequence

        seq_idx = 0
        for row in team_actions.itertuples():
            card_val = getattr(row, "card")
            if "ability" not in str(card_val):
                seq_idx += 1
            deck = place_sequence[seq_idx : seq_idx + 4]
            hand = [card for card in team_0_deck_list if card and card not in deck]
            while len(hand) < 4:
                hand.append("")
            for i in range(4):
                actions.loc[row.Index, f"hand_{i}"] = hand[i]

        # Approximate hand state for opponent_0 over time based on played cards.
        opp_actions = actions.loc[~is_team_0]
        opp_place_sequence = list(opp_actions["card"])
        opp_place_sequence = [action for action in opp_place_sequence if "ability" not in str(action)]
        opp_nonempty = [c for c in opponent_0_deck_list if c]
        opp_start_4 = random.sample(opp_nonempty, min(4, len(opp_nonempty))) if opp_nonempty else []
        while len(opp_start_4) < 4:
            opp_start_4.append("")
        opp_place_sequence = opp_start_4 + opp_place_sequence

        opp_seq_idx = 0
        for row in opp_actions.itertuples():
            card_val = getattr(row, "card")
            if "ability" not in str(card_val):
                opp_seq_idx += 1
            deck = opp_place_sequence[opp_seq_idx : opp_seq_idx + 4]
            hand = [card for card in opponent_0_deck_list if card and card not in deck]
            while len(hand) < 4:
                hand.append("")
            for i in range(4):
                actions.loc[row.Index, f"hand_{i}"] = hand[i]

        for i in range(8):
            actions[f"deck_{i}"] = opponent_0_deck_list[i]
            # Use team_actions.index instead of boolean mask to avoid indexer issues
            actions.loc[team_actions.index, f"deck_{i}"] = team_0_deck_list[i]
        
        traj_chunks.append(actions)

    traj_data = pd.concat(traj_chunks, ignore_index=True)
    traj_data.to_csv(save_path, index=False)


def build_traj_csv_from_placements(
    placements_path: str | list[str] = okezue_placements,
    save_path: str = save_csv,
    max_battles: int | None = None,
    chunk_size: int = 50_000,
    skip_ability: bool = True,
    skip_invalid: bool = True,
    excluded_cards: frozenset | set | None = EXCLUDED_CARDS,
):
    """
    Build traj CSV from placement-only data (no meta). Infers deck per battle per side
    from order of first card appearance; approximates hand with same sliding-window logic.
    Expects CSV with: battle_id, x, y, card, time, side, team, player_id (e.g. okezue format).
    Battles that contain any card in excluded_cards are skipped.
    If placements_path is a list, processes each file in order and appends to save_path.
    Writes to disk every chunk_size battles to limit memory use.
    """
    paths = [placements_path] if isinstance(placements_path, str) else list(placements_path)
    required = {"battle_id", "x", "y", "card", "time", "side", "team", "player_id"}
    traj_chunks = []
    total_battles_processed = 0
    total_battles_written = 0
    total_rows = 0
    header_written = False

    def flush_chunk():
        nonlocal traj_chunks, total_rows, total_battles_written, header_written
        if not traj_chunks:
            return
        out = pd.concat(traj_chunks, ignore_index=True)
        out.to_csv(save_path, index=False, mode="w" if not header_written else "a", header=not header_written)
        header_written = True
        total_rows += len(out)
        total_battles_written += len(traj_chunks)
        traj_chunks.clear()

    for path in paths:
        df = pd.read_csv(path)
        if not required.issubset(df.columns):
            raise ValueError(f"Placements CSV must have columns {required}; got {list(df.columns)}")
        if skip_invalid:
            df = df[df["card"].astype(str).str.strip() != "_invalid"]
            df = df[df["x"].notna() & df["y"].notna()]
        if skip_ability and "ability" in df.columns:
            df = df[df["ability"] != 1]
        df = df.sort_values(["battle_id", "time"])
        groups = list(df.groupby("battle_id", sort=False))
        if max_battles is not None:
            remaining = max_battles - total_battles_processed
            if remaining <= 0:
                break
            groups = groups[:remaining]
        for _battle_id, actions in groups:
            if total_battles_processed > 0 and total_battles_processed % 1000 == 0:
                print(f"{datetime.now()} at {total_battles_processed} battles processed")
            total_battles_processed += 1
            actions = actions.copy()
            actions = actions.reset_index(drop=True)
            if len(actions) < 2:
                continue
            if _battle_has_excluded_cards(actions, excluded_cards):
                continue
            is_team_0 = actions["side"].astype(str).str.strip() == "t"
            team_actions = actions.loc[is_team_0]
            opp_actions = actions.loc[~is_team_0]
            team_0_deck_list = _infer_deck_from_plays(team_actions["card"], skip_ability=skip_ability)
            opponent_0_deck_list = _infer_deck_from_plays(opp_actions["card"], skip_ability=skip_ability)
            actions["card"] = _normalize_card_series(actions["card"])

            place_sequence = list(team_actions["card"])
            place_sequence = [c for c in place_sequence if c and "ability" not in str(c)]
            team_deck_nonempty = [c for c in team_0_deck_list if c]
            for i in range(4):
                available = list(set(team_deck_nonempty) - set(place_sequence[:4]))
                one = random.sample(available, 1)[0] if available else ""
                place_sequence = [one] + place_sequence
            seq_idx = 0
            for row in team_actions.itertuples():
                card_val = getattr(row, "card", None)
                if card_val and "ability" not in str(card_val):
                    seq_idx += 1
                deck = place_sequence[seq_idx : seq_idx + 4] if seq_idx + 4 <= len(place_sequence) else place_sequence[-4:]
                hand = [c for c in team_0_deck_list if c and c not in deck]
                while len(hand) < 4:
                    hand.append("")
                for i in range(4):
                    actions.loc[row.Index, f"hand_{i}"] = hand[i]
            opp_place_sequence = list(opp_actions["card"])
            opp_place_sequence = [c for c in opp_place_sequence if c and "ability" not in str(c)]
            opp_deck_nonempty = [c for c in opponent_0_deck_list if c]
            for i in range(4):
                available = list(set(opp_deck_nonempty) - set(opp_place_sequence[:4]))
                one = random.sample(available, 1)[0] if available else ""
                opp_place_sequence = [one] + opp_place_sequence
            opp_seq_idx = 0
            for row in opp_actions.itertuples():
                card_val = getattr(row, "card", None)
                if card_val and "ability" not in str(card_val):
                    opp_seq_idx += 1
                deck = opp_place_sequence[opp_seq_idx : opp_seq_idx + 4] if opp_seq_idx + 4 <= len(opp_place_sequence) else opp_place_sequence[-4:]
                hand = [c for c in opponent_0_deck_list if c and c not in deck]
                while len(hand) < 4:
                    hand.append("")
                for i in range(4):
                    actions.loc[row.Index, f"hand_{i}"] = hand[i]
            for i in range(8):
                actions[f"deck_{i}"] = opponent_0_deck_list[i]
                actions.loc[team_actions.index, f"deck_{i}"] = team_0_deck_list[i]
            traj_chunks.append(actions)
            if len(traj_chunks) >= chunk_size:
                flush_chunk()
                print(f"  Wrote chunk ({chunk_size} battles) to {save_path}")
        if max_battles is not None and total_battles_processed >= max_battles:
            break

    flush_chunk()
    if total_rows == 0:
        print("No battles produced.")
        return
    print(f"Wrote {total_rows} rows to {save_path} ({total_battles_written} battles from {len(paths)} file(s)).")


def build_small_traj_csv(max_battles: int = 1000):
    """Convenience helper to build a smaller traj CSV for testing."""
    small_save_path = save_csv.replace(".csv", f"_small_{max_battles}.csv")
    build_traj_csv(save_path=small_save_path, max_battles=max_battles)
    # need to recreate each hand info. For 

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "placements":
        # Build from okezue placement files (part1 then part2), save in 50k-battle chunks.
        print("Building from okezue (part1 then part2)")
        build_traj_csv_from_placements(
            placements_path=[okezue_placements, okezue_placements_part2],
            save_path=save_csv.replace(".csv", "_okezue.csv"),
            chunk_size=50_000,
        )
    else:
        build_traj_csv()

