import pytest
from unittest.mock import patch, MagicMock
import yaml
import sys


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def _get_build_model():
    """Import build_model without triggering module-level build_model(cfg) call."""
    # Remove cached module so we can re-import cleanly each test
    if "api.server" in sys.modules:
        del sys.modules["api.server"]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Patch at transformers level so module-level build_model(cfg) call doesn't
    # attempt to load real model weights on disk.
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.GPTQConfig"):
        import api.server as server_module

    return server_module.build_model


def test_build_model_no_quant():
    """QUANT_BITS=0 时使用原始模型路径，无 GPTQConfig"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 0

    build_model = _get_build_model()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        tokenizer, model = build_model(cfg)

        args, kwargs = mock_load.call_args
        assert args[0] == cfg["MODEL_PATH"]
        assert "quantization_config" not in kwargs


def test_build_model_int4():
    """QUANT_BITS=4 时使用量化模型路径和 GPTQConfig(bits=4)"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 4
    cfg["QUANT_MODEL_PATH"] = "./models/qwen/Qwen-7B-Chat-GPTQ-int4"

    build_model = _get_build_model()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("api.server.GPTQConfig") as mock_gptq_cfg:

        tokenizer, model = build_model(cfg)

        mock_gptq_cfg.assert_called_once_with(bits=4, disable_exllama=False)
        args, kwargs = mock_load.call_args
        assert args[0] == cfg["QUANT_MODEL_PATH"]
        assert "quantization_config" in kwargs


def test_build_model_int8():
    """QUANT_BITS=8 时使用量化模型路径和 GPTQConfig(bits=8)"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 8
    cfg["QUANT_MODEL_PATH"] = "./models/qwen/Qwen-7B-Chat-GPTQ-int8"

    build_model = _get_build_model()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("api.server.GPTQConfig") as mock_gptq_cfg:

        tokenizer, model = build_model(cfg)

        mock_gptq_cfg.assert_called_once_with(bits=8, disable_exllama=False)
        args, kwargs = mock_load.call_args
        assert args[0] == cfg["QUANT_MODEL_PATH"]
        assert "quantization_config" in kwargs
