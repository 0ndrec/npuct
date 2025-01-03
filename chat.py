from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig, int4, int8
import torch
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def compile_model(model_id, dtype):
    # Compile for NPU acceleration using Intel NPU Acceleration Library
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True, weights_only=True).eval()
    torch_dtype = torch.int8 if dtype == "int8" else torch.float16
    with torch.no_grad():
        compiler_conf = CompilerConfig(dtype=torch_dtype)
        model = intel_npu_acceleration_library.compile(model, compiler_conf)
    return model


def load_model(model_id, dtype):
    PATH = os.path.join("models", model_id, dtype)
    filename = os.path.join(PATH, "model.pth")
    os.makedirs(PATH, exist_ok=True)

    if not os.path.exists(filename):
        model = compile_model(model_id, dtype)
        torch.save(model, filename)
    else:
        model = torch.load(filename).eval()
    return model


def generate_response(model, tokenizer, query):
    """Generate response using model and tokenizer"""
    try:
        messages = [
            {
                "role": "system",
                "content": "",
            },
            {"role": "user", "content": query},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            streamer=TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True),
        )

        return tokenizer.decode(outputs[0])
    except Exception as e:
        print(f"Error generating text: {e}")
        return None


if __name__ == "__main__":

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype = "int8"

    model = load_model(model_id, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    while True:
        query = input("> ")
        response = generate_response(model, tokenizer, query)
        if response is not None:
            print(response)

