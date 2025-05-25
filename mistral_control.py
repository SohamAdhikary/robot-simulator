# mistral_control.py
from ollama import Client
import time

class MistralController:
    def __init__(self):
        self.client = Client(host='http://localhost:11434')
        self.temperature = 0.2  # More deterministic
        self.max_retries = 3
        self.timeout = 30  # Retained for future use or manual handling

    def ask(self, prompt):
        """Robust query with retries and fallback"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model='mistral',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': self.temperature,
                        'num_ctx': 2048  # Larger context window
                    }
                    # Removed timeout parameter
                )
                content = response['message']['content'].strip()
                
                # Basic validation
                if len(content) > 0:
                    return content
                    
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return "no_action"  # Safe fallback

# Global instance for easy access
mistral = MistralController()

def ask_mistral(prompt):
    """Public interface for Mistral queries"""
    return mistral.ask(prompt)

if __name__ == "__main__":
    # Test the controller
    test_prompt = "Industrial robot in Move phase. Box at [0.6,0,0.5]. Target at [0.8,0,0.75]. Next action?"
    print("Testing Mistral controller...")
    print("Response:", ask_mistral(test_prompt))
