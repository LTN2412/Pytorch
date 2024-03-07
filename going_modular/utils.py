import torch
from pathlib import Path


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith('pth') or model_name.endswith(
        'pt'), "model_name should end with 'pth' or 'pt'"
    model_save_path = target_dir / model_name
    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(model.state_dict(), model_save_path)
