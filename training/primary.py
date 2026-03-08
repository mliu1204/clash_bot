import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from traj_dataloader import (
    DEFAULT_TRAJ_PATH,
    TrajDataset,
    TrajMode,
    pad_collate,
)


# Traj input: 17 features per step (x_norm, y_norm, time_norm, side, hand_0..3, deck_0..7, card)
TRAJ_FEAT_SIZE = 17


class TrajLSTM(nn.Module):
    """
    LSTM over variable-length traj sequences; predicts next (x, y) and card.
    Uses last hidden state (at true sequence end) to predict one target per sample.
    """

    def __init__(
        self,
        input_size: int = TRAJ_FEAT_SIZE,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_cards: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cards = num_cards
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_xy = nn.Linear(hidden_size, 3)  # x, y, time
        self.fc_card = nn.Linear(hidden_size, num_cards)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, max_len, input_size), lengths: (B,) on same device as x.
        Returns pred_xy (B, 3) [x, y, time], pred_card (B, num_cards).
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # Last valid output for each sample
        B = x.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_out = outputs[torch.arange(B, device=x.device), idx, :]
        pred_xy = self.fc_xy(last_out)
        pred_card = self.fc_card(last_out)
        return pred_xy, pred_card


def _evaluate(model: nn.Module, dataloader: DataLoader, criterion_xy: nn.Module, criterion_card: nn.Module, device: torch.device):
    """Run model on dataloader without grad; return avg loss_xy, loss_card, card accuracy."""
    model.eval()
    total_xy = 0.0
    total_card = 0.0
    n_samples = 0
    correct_card = 0
    with torch.no_grad():
        for x, lengths, target_xy, target_card in dataloader:
            x = x.to(device, dtype=torch.float32)
            lengths = lengths.to(device)
            target_xy = target_xy.to(device, dtype=torch.float32)
            target_card = target_card.to(device)
            pred_xy, pred_card = model(x, lengths)
            loss_xy = criterion_xy(pred_xy, target_xy)
            loss_card = criterion_card(pred_card, target_card)
            b = x.size(0)
            total_xy += loss_xy.item() * b
            total_card += loss_card.item() * b
            correct_card += (pred_card.argmax(dim=1) == target_card).sum().item()
            n_samples += b
    model.train()
    if n_samples == 0:
        return 0.0, 0.0, 0.0
    return total_xy / n_samples, total_card / n_samples, correct_card / n_samples


def train_traj_lstm(
    csv_path: str | None = None,
    mode: TrajMode = "both",
    num_epochs: int = 10,
    batch_size: int = 32,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 1e-3,
    skip_ability: bool = False,
    val_frac: float = 0.2,
    max_battle_count: int | None = None,
):
    if csv_path is None:
        csv_path = DEFAULT_TRAJ_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = TrajDataset(
        csv_path=csv_path,
        skip_ability=skip_ability,
        mode=mode,
        max_battle_count=max_battle_count,
    )
    n = len(full_dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )
    dataset = full_dataset  # for get_num_cards and feat_size
    num_cards = dataset.get_num_cards()
    # Infer input size from data so we always match the dataset (e.g. 17 features)
    feat_size = dataset[0][0].size(-1)

    model = TrajLSTM(
        input_size=feat_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_cards=num_cards,
    ).to(device)
    criterion_xy = nn.MSELoss()
    criterion_card = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss_xy = 0.0
        epoch_loss_card = 0.0
        n_samples = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            unit="batch",
        )
        for x, lengths, target_xy, target_card in pbar:
            x = x.to(device, dtype=torch.float32)
            lengths = lengths.to(device)
            target_xy = target_xy.to(device, dtype=torch.float32)
            target_card = target_card.to(device)

            optimizer.zero_grad()
            pred_xy, pred_card = model(x, lengths)
            loss_xy = criterion_xy(pred_xy, target_xy)
            loss_card = criterion_card(pred_card, target_card)
            loss = loss_xy + loss_card
            if not torch.isfinite(loss).all():
                tqdm.write(
                    f"Skip batch (loss nan): loss_xy={loss_xy.item():.4f} loss_card={loss_card.item():.4f}"
                )
                pbar.set_postfix(loss_xy="nan", loss_card="nan")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            b = x.size(0)
            epoch_loss_xy += loss_xy.item() * b
            epoch_loss_card += loss_card.item() * b
            n_samples += b
            pbar.set_postfix(loss_xy=f"{loss_xy.item():.3f}", loss_card=f"{loss_card.item():.3f}")

        if n_samples > 0:
            train_xy = epoch_loss_xy / n_samples
            train_card = epoch_loss_card / n_samples
            val_xy, val_card, val_acc = _evaluate(model, val_loader, criterion_xy, criterion_card, device)
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"train loss_xy: {train_xy:.4f}  train loss_card: {train_card:.4f} | "
                f"val loss_xy: {val_xy:.4f}  val loss_card: {val_card:.4f}  val card_acc: {val_acc:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - no valid batches (all nan/skip)")

    return model


def test_saved_model(
    model_path: str,
    csv_path: str | None = None,
    mode: TrajMode = "both",
    num_examples: int = 3,
    hidden_size: int = 64,
    num_layers: int = 1,
    skip_ability: bool = False,
    max_battle_count: int | None = 500,
    seed: int = 42,
):
    """
    Load a saved model and print example trajectories with model predictions vs ground truth.
    Uses the dataset to get vocab and example sequences; runs the model and prints each
    trajectory step-by-step, then the predicted next (x, y, time, card) vs actual.
    """
    if csv_path is None:
        csv_path = DEFAULT_TRAJ_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajDataset(
        csv_path=csv_path,
        skip_ability=skip_ability,
        mode=mode,
        max_battle_count=max_battle_count,
    )
    num_cards = dataset.get_num_cards()
    feat_size = dataset[0][0].size(-1)
    model = TrajLSTM(
        input_size=feat_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_cards=num_cards,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng).tolist()[:num_examples]
    if not indices:
        print("No examples in dataset.")
        return

    for ex_idx, idx in enumerate(indices):
        seq, length, target_x, target_y, target_time, target_card = dataset[idx]
        # seq: (L, 17) = [x, y, time, side, hand_0..3, deck_0..7, card] per step
        L = seq.size(0)
        print(f"\n{'='*60}")
        print(f"Example {ex_idx + 1}/{num_examples}  (seq length = {L})")
        print(f"{'='*60}")
        print("Trajectory (from current perspective: team=1, opponent=0):")
        for i in range(L):
            x, y, t, side = seq[i, 0].item(), seq[i, 1].item(), seq[i, 2].item(), seq[i, 3].item()
            card_idx = int(seq[i, 16].item())
            card_name = dataset.get_card_name(card_idx)
            side_str = "team" if side > 0.5 else "opponent"
            print(f"  step {i+1}: time={t:.4f}  side={side_str}  card={card_name}  x={x:.4f}  y={y:.4f}")

        # Run model on this single sample
        x_batch = seq.unsqueeze(0).to(device, dtype=torch.float32)
        lengths_batch = length.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_xy, pred_card = model(x_batch, lengths_batch)
        pred_x = pred_xy[0, 0].item()
        pred_y = pred_xy[0, 1].item()
        pred_time = pred_xy[0, 2].item()
        pred_card_idx = pred_card[0].argmax().item()
        pred_card_name = dataset.get_card_name(pred_card_idx)

        true_x = target_x.item() if torch.is_tensor(target_x) else float(target_x)
        true_y = target_y.item() if torch.is_tensor(target_y) else float(target_y)
        true_time = target_time.item() if torch.is_tensor(target_time) else float(target_time)
        true_card_idx = target_card.item() if torch.is_tensor(target_card) else int(target_card)
        true_card_name = dataset.get_card_name(true_card_idx)

        print("\nGround truth next move:")
        print(f"  x={true_x:.4f}  y={true_y:.4f}  time={true_time:.4f}  card={true_card_name}")
        print("Model predicts next move:")
        print(f"  x={pred_x:.4f}  y={pred_y:.4f}  time={pred_time:.4f}  card={pred_card_name}")
        print()

    print(f"{'='*60}\nDone.")


if __name__ == "__main__":
    # planner_model = train_traj_lstm(
    #     mode="planner",
    #     num_epochs=5,
    #     batch_size=32,
    #     hidden_size=64,
    #     num_layers=5,
    #     max_battle_count=1000,
    #     val_frac=0.15,
    # )
    # torch.save(planner_model.state_dict(), "planner_model.pt")
    # Test the saved planner model on a few example trajectories:
    test_saved_model("planner_model.pt", mode="planner", num_examples=3, hidden_size=64, num_layers=5, max_battle_count=500)
    # reacter_model = train_traj_lstm(
    #     mode="reacter",
    #     num_epochs=5,
    #     batch_size=32,
    #     hidden_size=64,
    #     num_layers=1,
    # )
