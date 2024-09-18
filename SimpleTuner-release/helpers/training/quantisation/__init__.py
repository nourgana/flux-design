from helpers.training.multi_process import should_log
import logging
import torch, os

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)

try:
    from optimum.quanto import freeze, quantize, qfloat8, qint8, qint4, qint2, QTensor
except ImportError as e:
    raise ImportError(
        f"To use Quanto, please install the optimum library: `pip install optimum-quanto`: {e}"
    )


def _quanto_model(model, model_precision, base_model_precision=None):
    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return

    logger.info(f"Quantising {model.__class__.__name__}. Using {model_precision}.")
    if model_precision == "int2-quanto":
        weight_quant = qint2
    elif model_precision == "int4-quanto":
        if torch.cuda.is_available():
            logger.error(
                "int4-quanto is only supported on A100 and H100 GPUs, but other GPUs would support int2-quanto, int8-quanto or fp8-quanto... waiting 10 seconds for you to cancel."
            )
            import time

            time.sleep(10)
        weight_quant = qint4
    elif model_precision == "int8-quanto":
        weight_quant = qint8
    elif model_precision == "fp8-quanto":
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS doesn't support dtype float8_e4m3n, you must select another precision level such as bf16, int2, int8, or int8."
            )

            return
        weight_quant = qfloat8
    else:
        raise ValueError(f"Invalid quantisation level: {base_model_precision}")
    quantize(model, weights=weight_quant)
    logger.info("Freezing model.")
    freeze(model)


def quantoise(
    unet, transformer, text_encoder_1, text_encoder_2, text_encoder_3, controlnet, args
):
    logger.info("Loading Quanto for LoRA training. This may take a few minutes.")
    if transformer is not None:
        _quanto_model(transformer, args.base_model_precision)
    if unet is not None:
        _quanto_model(unet, args.base_model_precision)
    if controlnet is not None:
        _quanto_model(controlnet, args.base_model_precision)

    if text_encoder_1 is not None:
        _quanto_model(
            text_encoder_1, args.text_encoder_1_precision, args.base_model_precision
        )
    if text_encoder_2 is not None:
        _quanto_model(
            text_encoder_2, args.text_encoder_2_precision, args.base_model_precision
        )
    if text_encoder_3 is not None:
        _quanto_model(
            text_encoder_3, args.text_encoder_3_precision, args.base_model_precision
        )
