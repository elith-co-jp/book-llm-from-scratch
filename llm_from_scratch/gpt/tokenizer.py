"""Simple character-level tokenizer for GPT."""


class SimpleTokenizer:
    """Character-level tokenizer for text processing."""
    
    def __init__(self, text):
        """
        Initialize tokenizer with vocabulary from text.
        
        Args:
            text: Text corpus to build vocabulary from
        """
        # Build vocabulary from unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens):
        """
        Decode token IDs to text.
        
        Args:
            tokens: List or tensor of token IDs
            
        Returns:
            Decoded text string
        """
        return ''.join([self.idx_to_char.get(int(idx), '') for idx in tokens])