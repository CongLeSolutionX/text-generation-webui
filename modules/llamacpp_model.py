import re
from functools import partial
from pathlib import Path
from typing import Union

import torch

from modules import RoPE, shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length
from modules.utils import is_gguf

import llama_cpp

try:
    import llama_cpp_ggml
except:
    llama_cpp_ggml = llama_cpp

if torch.cuda.is_available() and not torch.version.hip:
    try:
        import llama_cpp_cuda
    except:
        llama_cpp_cuda = None
    try:
        import llama_cpp_ggml_cuda
    except:
        llama_cpp_ggml_cuda = llama_cpp_cuda
else:
    llama_cpp_cuda = None
    llama_cpp_ggml_cuda = None


def llama_cpp_lib(model_file: Union[str, Path] = None):
    gguf_model = is_gguf(model_file) if model_file is not None else True
    if shared.args.cpu or llama_cpp_cuda is None:
        return llama_cpp if gguf_model else llama_cpp_ggml
    else:
        return llama_cpp_cuda if gguf_model else llama_cpp_ggml_cuda


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    def __del__(self):
        self.model.__del__()

    @classmethod
    def from_pretrained(cls, path):

        Llama = llama_cpp_lib(path).Llama
        LlamaCache = llama_cpp_lib(path).LlamaCache

        result = cls()
        cache_capacity = 0
        if shared.args.cache_capacity is not None:
            if 'GiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000 * 1000
            elif 'MiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000
            else:
                cache_capacity = int(shared.args.cache_capacity)

        logger.info(f"Cache capacity is {cache_capacity} bytes")

        if shared.args.tensor_split is None or shared.args.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in shared.args.tensor_split.strip().split(",")]

        params = {
            'model_path': str(path),
            'n_ctx': shared.args.n_ctx,
            'seed': int(shared.args.llama_cpp_seed),
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'mul_mat_q': shared.args.mul_mat_q,
            'low_vram': shared.args.low_vram,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': RoPE.get_rope_freq_base(shared.args.alpha_value, shared.args.rope_freq_base),
            'tensor_split': tensor_split_list,
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
        }

        if not is_gguf(path):
            ggml_params = {
                'n_gqa': shared.args.n_gqa or None,
                'rms_norm_eps': shared.args.rms_norm_eps or None,
            }
            params |= ggml_params

        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, tokens):
        return self.model.detokenize(tokens)

    def generate(self, prompt, state, callback=None):

        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList

        prompt = prompt if type(prompt) is str else prompt.decode()

        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state):]
        prompt = self.decode(prompt).decode('utf-8')

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            repeat_penalty=state['repetition_penalty'],
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            stream=True,
            logits_processor=LogitsProcessorList([
                partial(ban_eos_logits_processor, self.model.token_eos()),
            ]) if state['ban_eos_token'] else None,
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break
            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
