import torch
from typing import List, Union, Generator, Optional
from transformers import AutoTokenizer

class TokenizerManager:
    """
    Manages loading and configuring the Hugging Face AutoTokenizer.
    """
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_tokenizer(self):
        return self.tokenizer
        
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
        
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
        
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


class TextEncoder:
    """
    Encodes raw string or batches of strings into subword token IDs.
    """
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.manager = tokenizer_manager
        self.tokenizer = tokenizer_manager.get_tokenizer()
        
    def encode(
        self, 
        text: Union[str, List[str]], 
        max_length: Optional[int] = None, 
        padding: Union[bool, str] = False, 
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], List[List[int]], torch.Tensor]:
        """
        Encodes text(s) into token ids.
        Args:
            text: Single string or list of strings.
            max_length: Optional max sequence length.
            padding: Padding strategy ('max_length', 'longest', or boolean).
            truncation: Whether to truncate to max_length.
            return_tensors: If 'pt', returns PyTorch Tensor.
        """
        # Call the underlying HF tokenizer
        kwargs = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
            
        outputs = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            **kwargs
        )
        
        if return_tensors == "pt":
            return outputs["input_ids"]
        return outputs["input_ids"]


class TextDecoder:
    """
    Decodes token IDs back into natural text, supporting standard batches and streaming.
    """
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.manager = tokenizer_manager
        self.tokenizer = tokenizer_manager.get_tokenizer()
        
    def decode(
        self, 
        token_ids: Union[int, List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decodes token ids to string(s).
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim == 1:
                return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)
            else:
                return self.tokenizer.batch_decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)
                
        if isinstance(token_ids, list):
            if len(token_ids) == 0:
                return ""
            if isinstance(token_ids[0], list):
                return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            
        # Single token ID
        return self.tokenizer.decode([token_ids], skip_special_tokens=skip_special_tokens)
        
    def decode_stream(
        self, 
        token_generator: Generator[Union[int, torch.Tensor], None, None],
        skip_special_tokens: bool = True
    ) -> Generator[str, None, None]:
        """
        Yields decoded text chunks on-the-fly from a token ID stream.
        Handles multi-byte characters and subword boundary decoding appropriately.
        """
        # Maintain a list of generated tokens to decode with correct subword context
        tokens_so_far = []
        decoded_text_length = 0
        
        for token in token_generator:
            if isinstance(token, torch.Tensor):
                token_val = token.item()
            else:
                token_val = token
                
            tokens_so_far.append(token_val)
            
            # Decode the cumulative sequence to handle trailing partial bytes/chars
            full_decoded = self.tokenizer.decode(tokens_so_far, skip_special_tokens=skip_special_tokens)
            
            # Extract only the newly added text chunk
            new_text = full_decoded[decoded_text_length:]
            decoded_text_length = len(full_decoded)
            
            yield new_text
