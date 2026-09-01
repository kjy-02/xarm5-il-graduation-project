import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from xarm_il.model import ActionChunkTransformer, ModelConfig, masked_action_loss  # noqa: E402


def test_model_forward_and_masked_loss_backward() -> None:
    config = ModelConfig(
        camera_count=1,
        state_dim=10,
        horizon=4,
        d_model=64,
        nhead=4,
        decoder_layers=1,
        dropout=0.0,
    )
    model = ActionChunkTransformer(config)
    images = torch.randn(2, 1, 3, 64, 64)
    state = torch.randn(2, 10)
    target = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
    prediction = model(images, state)
    loss, parts = masked_action_loss(prediction, target, mask)
    loss.backward()
    assert prediction.shape == (2, 4, 3)
    assert torch.isfinite(loss)
    assert set(parts) == {"imitation", "smoothness"}
