import argparse
from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
import intel_npu_acceleration_library
import torch

class ChatModel:
    def __init__(self):
        self.model_id = "meta-llama/Llama-2-7b-chat-hf"
        try:
            self.model = NPUModelForCausalLM.from_pretrained(self.model_id, use_cache=True).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_default_system_prompt=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
            self.model = intel_npu_acceleration_library.compile(self.model, dtype=torch.int8)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")

    def generate_response(self, query: str) -> str:
        prefix = self.tokenizer(query, return_tensors="pt")["input_ids"]
        generation_kwargs = dict(
            input_ids=prefix,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            max_new_tokens=512,
        )
        output = self.model.generate(**generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPU Powered Chat")
    parser.add_argument("query", type=str, help="Your query to the model")
    args = parser.parse_args()

    chat = ChatModel()
    response = chat.generate_response(args.query)
    print(response)
