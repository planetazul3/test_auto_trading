import torch

from src.models.cnn_extractor import CNN1DExtractor
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.tft_attention import TFTFusionNode
from src.models.hybrid_tft import HybridCNNLSTMTFT


def test_cnn_extractor_returns_sequence():
    model = CNN1DExtractor(num_features=12, sequence_length=20, embedding_dim=16, return_sequence=True)
    x = torch.randn(4, 20, 12)
    out = model(x)
    assert out.shape == (4, 20, 16)
    assert not torch.isnan(out).any()


def test_bilstm_encoder_collapses_when_not_returning_sequence():
    model = BiLSTMEncoder(input_size=12, hidden_size=16, embedding_dim=16)
    x = torch.randn(4, 20, 12)
    out = model(x)
    assert out.shape == (4, 16)


def test_tft_fusion_causal_mask_shape():
    fusion = TFTFusionNode(embedding_dim=16, num_heads=4, num_sources=2, output_dim=16)
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 8, 16)
    out, attn = fusion([a, b])
    assert out.shape == (2, 8, 16)
    # MultiheadAttention con average heads → (batch, seq, seq).
    assert attn.shape[0] == 2 and attn.shape[-1] == 8


def test_hybrid_emits_logits_and_attn():
    model = HybridCNNLSTMTFT(
        input_features=12,
        sequence_length=20,
        cnn_channels=8,
        lstm_hidden=16,
        tft_hidden=16,
        num_attention_heads=2,
        lstm_layers=1,
    )
    x = torch.randn(3, 20, 12)
    logits, attn = model(x)
    assert logits.shape == (3, 1)
    assert attn.numel() > 0
