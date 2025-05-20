import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from .calib_data import get_pile_calib_dataset, get_reasoning_calib_dataset


def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--model', type=str, default='./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B',
                        help='Model to load')
    parser.add_argument('--method', type=str, default='awq-autoawq',
                        choices=['awq-autoawq', 'awq-llmcompressor', 'gptq-gptqmodel', 'gptq-llmcompressor'],
                        help='Supported real-quantization methods')
    parser.add_argument("--w_bits", type=int, default=4,
                        choices=[4], help='#bits for weights')
    parser.add_argument("--w_groupsize", type=int, default=128, help='Weight quantization group size')
    parser.add_argument("--w_asym", action="store_true", help='Asymmetric weight quantization')
    args = parser.parse_args()

    args.model_name = args.model.split("/")[-1]
    args.save_qmodel_path = os.path.join(
        "./outputs/modelzoo/real_quantization", args.method,
        args.model.split("/")[-1] + f"-quantized.{args.method}-w{args.w_bits}g{args.w_groupsize}"
    )

    return args


if __name__ == "__main__":
    args = parser_gen()
    
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for connection...")
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    if args.method == "awq-autoawq":
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_pretrained(
            args.model, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        quant_config = {
            "zero_point": args.w_asym,
            "q_group_size": args.w_groupsize,
            "w_bit": args.w_bits,
            "version": "GEMM"
        }
        model.quantize(
            tokenizer=tokenizer,
            quant_config=quant_config,
            calib_data="./datasets/pile-val-backup",
            split="validation",
            text_column="text",
            max_calib_samples=128,
            max_calib_seq_len=512,
            duo_scaling=False,
            apply_clip=True,
        )

        model.save_quantized(args.save_qmodel_path)
        tokenizer.save_pretrained(args.save_qmodel_path)
    elif args.method == "gptq-gptqmodel":
        from gptqmodel import GPTQModel, QuantizeConfig

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        ds = get_reasoning_calib_dataset(model_name=args.model_name, tokenizer=tokenizer, n_samples=128, seqlen=2048,
                                         return_attention_mask=True)

        quant_config = QuantizeConfig(
            bits=args.w_bits,
            group_size=args.w_groupsize,
            sym=not args.w_asym,
            desc_act=True,
            static_groups=True,
            mse=2.4,
        )
        model = GPTQModel.load(args.model, quant_config)

        # increase `batch_size` to match gpu/vram specs to speed up quantization
        model.quantize(calibration_dataset=ds, batch_size=2)

        model.save(args.save_qmodel_path)
        tokenizer.save_pretrained(args.save_qmodel_path)
    elif args.method.split("-")[-1] == "llmcompressor":
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationScheme,
            QuantizationStrategy,
            QuantizationType,
        )
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier, AWQ_MAPPING_REGISTRY
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

        assert not args.w_asym  # only supports symmetric quantization

        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if args.method == "awq-llmcompressor":
            # Load dataset
            ds = get_pile_calib_dataset(tokenizer=tokenizer, n_samples=128, block_size=512)

            # Configure the quantization algorithm to run
            recipe = [
                AWQModifier(
                    bits=args.w_bits,
                    group_size=args.w_groupsize,
                    symmetric=not args.w_asym,
                    mappings=AWQ_MAPPING_REGISTRY["Llama"],
                    duo_scaling=False,
                ),
                QuantizationModifier(
                    config_groups={
                        "group_0": QuantizationScheme(
                            targets=["Linear"],
                            weights=QuantizationArgs(
                                num_bits=args.w_bits,
                                type=QuantizationType.INT,
                                dynamic=False,
                                symmetric=not args.w_asym,
                                strategy=QuantizationStrategy.GROUP,
                                group_size=args.w_groupsize,
                                # The weight clipping strategy is slightly different from AWQ paper.
                                # The clipping thresholds are determined based on quantization error
                                # of weights instead of outputs of the linear layer.
                                observer="mse",
                            ),
                        )
                    },
                    ignore=["lm_head"],
                ),
            ]
        elif args.method == "gptq-llmcompressor":
            # Load dataset
            ds = get_reasoning_calib_dataset(model_name=args.model_name, tokenizer=tokenizer, n_samples=128, seqlen=2048)

            # Configure the quantization algorithm to run
            recipe = GPTQModifier(
                config_groups={
                    "group_0": QuantizationScheme(
                        targets=["Linear"],
                        weights=QuantizationArgs(
                            num_bits=args.w_bits,
                            type=QuantizationType.INT,
                            dynamic=False,
                            symmetric=not args.w_asym,
                            strategy=QuantizationStrategy.GROUP,
                            group_size=args.w_groupsize,
                            actorder="weight",  # "weight" for better performance, "group" for better accuracy
                            observer="mse",
                        ),
                    ),
                },
                ignore=["lm_head"],
                dampening_frac=0.01,    # set to larger value (e.g. 0.1) if Hessian inversion failed
            )

        # Run calibration
        model = oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            shuffle_calibration_samples=False,
        )

        # Save compressed model
        model.save_pretrained(args.save_qmodel_path, save_compressed=True)
        tokenizer.save_pretrained(args.save_qmodel_path)
