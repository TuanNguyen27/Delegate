# tools/api_key_manager.py
"""
API Key Pool Manager - Rotate through multiple API keys to avoid rate limits
"""
import os
import time
from typing import List, Optional
import google.generativeai as genai


class APIKeyManager:
    """
    Manages a pool of API keys and rotates through them to avoid rate limits.
    
    Usage:
        manager = APIKeyManager(['key1', 'key2', 'key3'])
        model = manager.get_model()  # Uses next available key
    """
    
    def __init__(self, api_keys: List[str], cooldown_seconds: int = 5):
        """
        Initialize API key manager.
        
        Args:
            api_keys: List of API keys to rotate through
            cooldown_seconds: Time to wait before reusing a key (default: 60s)
        """
        self.api_keys = api_keys
        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.key_last_used = {}  # Track when each key was last used
        self.total_keys = len(api_keys)
        
        if self.total_keys == 0:
            raise ValueError("At least one API key must be provided")
        
        print(f"✓ API Key Manager initialized with {self.total_keys} keys")
    
    def get_next_key(self) -> str:
        """
        Get the next available API key, waiting if necessary.
        
        Returns:
            str: Next API key to use
        """
        attempts = 0
        max_attempts = self.total_keys * 2  # Avoid infinite loop
        
        while attempts < max_attempts:
            key = self.api_keys[self.current_index]
            last_used = self.key_last_used.get(self.current_index, 0)
            time_since_use = time.time() - last_used
            
            # If this key has cooled down, use it
            if time_since_use >= self.cooldown_seconds:
                self.key_last_used[self.current_index] = time.time()
                
                # Move to next key for next call
                self.current_index = (self.current_index + 1) % self.total_keys
                
                return key
            
            # Key still in cooldown, try next one
            self.current_index = (self.current_index + 1) % self.total_keys
            attempts += 1
            
            # If we've tried all keys and all are in cooldown, wait a bit
            if attempts % self.total_keys == 0:
                wait_time = self.cooldown_seconds - time_since_use + 1
                print(f"⏳ All keys in cooldown. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
        
        # Fallback: return current key even if in cooldown
        return self.api_keys[self.current_index]
    
    def configure_next_key(self):
        """Configure genai with the next available API key."""
        key = self.get_next_key()
        genai.configure(api_key=key)
        print(f"🔑 Using API key #{self.current_index + 1}/{self.total_keys}")
    
    def get_model(self, model_name: str = "gemini-2.5-flash-lite", **kwargs):
        """
        Get a GenerativeModel configured with the next available API key.
        
        Args:
            model_name: Name of the Gemini model
            **kwargs: Additional arguments for GenerativeModel
        
        Returns:
            genai.GenerativeModel: Configured model instance
        """
        self.configure_next_key()
        return genai.GenerativeModel(model_name=model_name, **kwargs)


def load_api_keys_from_env() -> List[str]:
    """
    Load API keys from environment variables.
    
    Looks for:
    - GOOGLE_API_KEY (single key)
    - GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... (multiple keys)
    - GOOGLE_API_KEYS (comma-separated list)
    
    Returns:
        List[str]: List of API keys found
    """
    keys = []
    
    # Method 1: Single key
    single_key = os.getenv('GOOGLE_API_KEY')
    if single_key:
        keys.append(single_key)
    
    # Method 2: Numbered keys (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.)
    i = 1
    while True:
        key = os.getenv(f'GOOGLE_API_KEY_{i}')
        if key:
            keys.append(key)
            i += 1
        else:
            break
    
    # Method 3: Comma-separated list
    keys_list = os.getenv('GOOGLE_API_KEYS')
    if keys_list:
        keys.extend([k.strip() for k in keys_list.split(',') if k.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    
    return unique_keys


def load_api_keys_from_kaggle() -> List[str]:
    """
    Load API keys from Kaggle secrets.
    
    Looks for:
    - GOOGLE_API_KEY
    - GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
    
    Returns:
        List[str]: List of API keys found
    """
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        keys = []
        
        # Try single key first
        try:
            key = user_secrets.get_secret("GOOGLE_API_KEY")
            keys.append(key)
        except:
            pass
        
        # Try numbered keys
        i = 1
        while i <= 10:  # Try up to 10 keys
            try:
                key = user_secrets.get_secret(f"GOOGLE_API_KEY_{i}")
                keys.append(key)
                i += 1
            except:
                break
        
        return keys
    except ImportError:
        # Not running on Kaggle
        return []


def create_key_manager(cooldown_seconds: int = 1) -> APIKeyManager:
    """
    Create an API key manager with keys from environment or Kaggle.
    
    Args:
        cooldown_seconds: Cooldown time between key reuse
    
    Returns:
        APIKeyManager: Configured key manager
    """
    # Try Kaggle first
    keys = load_api_keys_from_kaggle()
    
    # Fall back to environment variables
    if not keys:
        keys = load_api_keys_from_env()
    
    if not keys:
        raise ValueError(
            "No API keys found. Please set:\n"
            "  - GOOGLE_API_KEY (single key), or\n"
            "  - GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... (multiple keys), or\n"
            "  - Add them as Kaggle secrets"
        )
    
    print(f"📊 Loaded {len(keys)} API key(s)")
    return APIKeyManager(keys, cooldown_seconds=cooldown_seconds)


# Example usage
if __name__ == "__main__":
    # Example 1: Direct initialization
    keys = ["key1", "key2", "key3"]
    manager = APIKeyManager(keys, cooldown_seconds=1)
    
    # Get a model
    model = manager.get_model("gemini-2.5-flash-lite")
    
    # Example 2: Load from environment
    manager = create_key_manager()
    model = manager.get_model()

