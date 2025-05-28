from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MedicalSummarizer:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.model_name = "facebook/bart-large-cnn"  # Using BART model which works well for summarization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=self.model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Set generation parameters
        self.max_length = 130
        self.min_length = 30
        self.do_sample = False

    def summarize_text(self, text: str) -> str:
        """
        Summarize medical text using a pre-trained model
        
        Args:
            text (str): Input medical text to be summarized
            
        Returns:
            str: Summarized text
        """
        try:
            # Split text into chunks if it's too long (BART has a max length of 1024 tokens)
            chunks = self._chunk_text(text)
            
            # Generate summary for each chunk and combine
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    do_sample=self.do_sample,
                    truncation=True
                )
                summaries.append(summary[0]['summary_text'])
            
            return " ".join(summaries)
            
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return "Error occurred during summarization"
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000) -> list:
        """
        Split text into chunks of approximately equal size
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'summarizer'):
            del self.summarizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
