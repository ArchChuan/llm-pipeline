import pytest
import os
import yaml
from unittest.mock import patch, MagicMock


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def test_config_has_quant_fields():
    cfg = load_cfg()
    assert "QUANT_BITS" in cfg
    assert "QUANT_MODEL_PATH" in cfg
    assert "QUANT_SAMPLES" in cfg
    assert cfg["QUANT_BITS"] == 0  # 默认不量化


def test_quant_bits_zero_skips_quantization():
    """QUANT_BITS=0 时 should_quantize 返回 False"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 0
    from scripts.quantize import should_quantize
    assert should_quantize(cfg) is False


def test_quant_bits_4_triggers_quantization():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 4
    from scripts.quantize import should_quantize
    assert should_quantize(cfg) is True


def test_quant_bits_8_triggers_quantization():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 8
    from scripts.quantize import should_quantize
    assert should_quantize(cfg) is True


def test_invalid_quant_bits_raises():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 3
    from scripts.quantize import should_quantize
    with pytest.raises(ValueError, match="QUANT_BITS must be 0, 4, or 8"):
        should_quantize(cfg)
