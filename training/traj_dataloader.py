"""
Data loader for traj.csv: battle placement sequences with hand/deck context.
Each battle appears twice: once from team's perspective, once from opponent's.
Side is always encoded as team=1 / opponent=0 for the current perspective; the target
is always the next action of the current side (team), so the model always predicts "team" move.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TrajMode = Literal["planner", "reacter", "both"]

DEFAULT_TRAJ_PATH = Path(__file__).resolve().parents[1] / "data" / "ready_data" / "traj.csv"
PAD_IDX = 0  # same as vocab["<pad>"]; used to pad variable-length sequences


def build_card_vocab(df: pd.DataFrame) -> dict[str, int]:
    """Build card name -> index from card, hand_*, and deck_* columns. Index 0 = padding/unknown."""
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    cols = [c for c in (["card"] + [f"hand_{i}" for i in range(4)] + [f"deck_{i}" for i in range(8)]) if c in df.columns]
    if not cols:
        return vocab
    # Single pass over all unique values
    uniq = pd.unique(df[cols].astype(str).values.ravel())
    for name in uniq:
        name = (name or "").strip()
        if name and name != "nan" and name not in vocab:
            vocab[name] = len(vocab)
    return vocab


def encode_card(vocab: dict[str, int], name: str) -> int:
    return vocab.get(str(name).strip() if pd.notna(name) else "", vocab.get("<unk>", 1))


def _encode_column(vocab: dict[str, int], col: pd.Series) -> np.ndarray:
    """Vectorized encode of a column of card names to indices."""
    return col.map(lambda x: encode_card(vocab, x)).to_numpy(dtype=np.int64)


def pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences; return (x_padded, lengths, target_xy, target_card). target_xy is (B, 3): [x, y, time]."""
    seqs, lengths, tx, ty, ttime, tcard = zip(*batch)
    lengths_t = torch.stack(lengths)
    padded = pad_sequence(seqs, batch_first=True, padding_value=float(PAD_IDX))
    target_xy = torch.stack((torch.stack(tx), torch.stack(ty), torch.stack(ttime)), dim=1)
    target_card = torch.stack(tcard)
    return padded, lengths_t, target_xy, target_card


class TrajDataset(Dataset):
    """
    Dataset over traj.csv: each battle appears twice (team perspective, opponent perspective).
    Side in features is always team=1 / opponent=0 for the current perspective.
    Target is always the next action of the current side (team): (x, y, time, card).

    Modes:
      - "planner": only (team → team): last move in sequence was team, next move (target) is team.
      - "reacter": only (opponent → team): last move in sequence was opponent, next move (target) is team.
      - "both": include all samples where target is team (no filter on who made the last move).
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_TRAJ_PATH,
        skip_ability: bool = True,
        mode: TrajMode = "both",
        max_battle_count: int | None = None,
    ):
        """
        Args:
            csv_path: Path to traj.csv.
            skip_ability: If True, drop rows where card contains "ability".
            mode: "planner" (team→team), "reacter" (opponent→team), or "both".
            max_battle_count: If set, only process this many battles (for faster testing). None = use all.
        """
        self.csv_path = Path(csv_path)
        self.skip_ability = skip_ability
        self.mode = mode
        self.max_battle_count = max_battle_count

        df = pd.read_csv(self.csv_path)
        # remove the ones that have na in card name entry
        df = df.groupby("battle_id").filter(lambda g: g["card"].notna().all())
        df["x"] = df["x"].fillna(499.000000)
        df["y"] = df["y"].fillna(499.000000)

        if skip_ability:
            df = df[~df["card"].astype(str).str.contains("ability", na=False)]

        self.vocab = build_card_vocab(df)
        self.idx_to_card: dict[int, str] = {idx: name for name, idx in self.vocab.items()}
        self.num_cards = len(self.vocab)
        n_battles = df.battle_id.nunique()
        print(f"Total number of battles: {n_battles}")

        df = df.sort_values(["battle_id", "time"])
        df.x = (df.x - 499.000000)/(17500.000000-499.000000)
        df.y = (df.y - 499.000000)/(31500.000000-499.000000)
        df.time = df.time/6000.0
        groups = list(df.groupby("battle_id", sort=False))

        self.samples: list[tuple[torch.Tensor, float, float, float, int]] = []
        x_col = "x"
        y_col = "y"
        time_col = "time"
        side_col = "side"
        hand_cols = [f"hand_{i}" for i in range(4)]
        deck_cols = [f"deck_{i}" for i in range(8)]

        # Raw CSV: side "t" = team (blue), "o" = opponent (red)
        for i, (_battle_id, grp) in enumerate(groups):
            if self.max_battle_count is not None and i >= self.max_battle_count:
                break
            if i % 500 == 0 and i > 0:
                print(f"  {i} / {min(n_battles, self.max_battle_count or n_battles)} battles")
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue

            # Precompute normalized numerics and indices for the whole battle (vectorized)

            n_rows = len(grp)
            side_is_t = (grp[side_col].astype(str).str.strip() == "t").to_numpy()

            card_idx = _encode_column(self.vocab, grp["card"])
            hand_idxs = np.column_stack([_encode_column(self.vocab, grp[c]) for c in hand_cols])
            deck_idxs = np.column_stack([_encode_column(self.vocab, grp[c]) for c in deck_cols])

            # Two perspectives: team (raw "t") and opponent (raw "o"). Side enc: 1 = current perspective side.
            x_vals = grp[x_col].to_numpy(dtype=np.float64)
            y_vals = grp[y_col].to_numpy(dtype=np.float64)
            time_vals = grp[time_col].to_numpy(dtype=np.float64)

            for raw_team in ("t", "o"):
                side_enc = side_is_t.astype(np.float32) if raw_team == "t" else (~side_is_t).astype(np.float32)
                target_ok = side_is_t if raw_team == "t" else ~side_is_t  # target row must be this side
                if raw_team == "o":
                    x_feat = (1.0 - x_vals).astype(np.float32)
                    y_feat = (1.0 - y_vals).astype(np.float32)
                else:
                    x_feat = (x_vals).astype(np.float32)
                    y_feat = (y_vals).astype(np.float32)

                for t in range(1, n_rows):
                    if not target_ok[t]:
                        continue  # target must be current side's next move
                    last_move_was_team = side_enc[t - 1] > 0.5
                    if self.mode == "planner" and not last_move_was_team:
                        continue
                    if self.mode == "reacter" and last_move_was_team:
                        continue
                    # Build step matrix for indices [0:t] in one go (no iterrows)
                    sl = slice(0, t)
                    steps = np.concatenate([
                        x_feat[sl, np.newaxis],
                        y_feat[sl, np.newaxis],
                        time_vals[sl, np.newaxis].astype(np.float32),
                        side_enc[sl, np.newaxis],
                        hand_idxs[sl].astype(np.float32),
                        deck_idxs[sl].astype(np.float32),
                        card_idx[sl, np.newaxis].astype(np.float32),
                    ], axis=1)
                    seq_tensor = torch.from_numpy(steps)
                    raw_tx = float(grp.iloc[t][x_col])
                    raw_ty = float(grp.iloc[t][y_col])
                    target_x = (1.0 - raw_tx) if raw_team == "o" else raw_tx
                    target_y = (1.0 - raw_ty) if raw_team == "o" else raw_ty
                    target_time = float(time_vals[t])
                    target_card_idx = int(card_idx[t])
                    self.samples.append((seq_tensor, target_x, target_y, target_time, target_card_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, target_x, target_y, target_time, target_card_idx = self.samples[idx]
        length = torch.tensor(seq.size(0), dtype=torch.long)
        return (
            seq,
            length,
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_y, dtype=torch.float32),
            torch.tensor(target_time, dtype=torch.float32),
            torch.tensor(target_card_idx, dtype=torch.long),
        )

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.copy()

    def get_num_cards(self) -> int:
        return self.num_cards

    def get_card_name(self, idx: int) -> str:
        """Decode card index to name (e.g. for logging)."""
        return self.idx_to_card.get(int(idx), "<unk>")


def get_traj_dataloader(
    csv_path: str | Path = DEFAULT_TRAJ_PATH,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    skip_ability: bool = True,
    mode: TrajMode = "both",
    max_battle_count: int | None = None,
):
    """Build TrajDataset and return a DataLoader. Each batch is (x, lengths, target_xy, target_card).
    mode: "planner" (team→team), "reacter" (opponent→team), "both". max_battle_count: cap battles for testing."""
    from torch.utils.data import DataLoader

    dataset = TrajDataset(
        csv_path=csv_path,
        skip_ability=skip_ability,
        mode=mode,
        max_battle_count=max_battle_count,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )


if __name__ == "__main__":
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader

    ds = TrajDataset(DEFAULT_TRAJ_PATH, skip_ability=True)
    print(f"TrajDataset: {len(ds)} samples, vocab size = {ds.get_num_cards()}")

    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=pad_collate)
    for x, lengths, target_xy, target_card in loader:
        print(f"x shape: {x.shape}, lengths: {lengths.tolist()}")
        print(f"target_xy shape: {target_xy.shape}, target_card shape: {target_card.shape}")
        print(f"first target: (x,y,time)=({target_xy[0,0].item():.4f}, {target_xy[0,1].item():.4f}, {target_xy[0,2].item():.4f}), card={ds.get_card_name(target_card[0].item())}")
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        break
    print("Data loader OK.")
