#!/usr/bin/env python3
"""
Quick API Key Checker - Verify your API key configuration
Run this to check if your 5 API keys are properly set up and all different
"""
import sys
import os
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("="*70)
    print("🔑 API KEY VERIFICATION")
    print("="*70)
    print()
    
    try:
        from tools.api_key_manager import load_api_keys_from_kaggle, load_api_keys_from_env
        
        # Try Kaggle first
        print("Checking Kaggle secrets...")
        keys = load_api_keys_from_kaggle()
        source = "Kaggle Secrets"
        
        if not keys:
            print("  Not running on Kaggle or no keys in secrets")
            print()
            print("Checking environment variables...")
            from dotenv import load_dotenv
            load_dotenv()
            keys = load_api_keys_from_env()
            source = "Environment Variables"
        
        if not keys:
            print()
            print("❌ NO API KEYS FOUND!")
            print()
            print("📝 How to add keys:")
            print()
            print("On Kaggle:")
            print("  1. Click Add-ons → Secrets")
            print("  2. Add:")
            print("     - GOOGLE_API_KEY_1 = your-first-key")
            print("     - GOOGLE_API_KEY_2 = your-second-key")
            print("     - GOOGLE_API_KEY_3 = your-third-key")
            print("     - GOOGLE_API_KEY_4 = your-fourth-key")
            print("     - GOOGLE_API_KEY_5 = your-fifth-key")
            print()
            print("Locally (.env file or environment):")
            print("  export GOOGLE_API_KEY_1=your-first-key")
            print("  export GOOGLE_API_KEY_2=your-second-key")
            print("  etc.")
            return 1
        
        # Display results
        print()
        print("="*70)
        print(f"✅ FOUND {len(keys)} API KEY(S) from {source}")
        print("="*70)
        print()
        
        # Show each key
        for i, key in enumerate(keys, 1):
            # Show first 10 and last 4 characters
            if len(key) >= 14:
                preview = f"{key[:10]}...{key[-4:]}"
            else:
                preview = f"{key[:6]}...{key[-2:]}"
            
            print(f"  Key #{i}: {preview} (length: {len(key)})")
        
        print()
        print("-"*70)
        
        # Check for duplicates
        unique_keys = set(keys)
        num_unique = len(unique_keys)
        num_total = len(keys)
        
        if num_unique < num_total:
            print()
            print("⚠️  WARNING: DUPLICATE KEYS DETECTED!")
            print()
            print(f"Total keys: {num_total}")
            print(f"Unique keys: {num_unique}")
            print(f"Duplicates: {num_total - num_unique}")
            print()
            print("Which keys are duplicated:")
            
            key_counts = Counter(keys)
            for j, (key, count) in enumerate(key_counts.items(), 1):
                if count > 1:
                    preview = f"{key[:10]}...{key[-4:]}"
                    print(f"  ❌ {preview} appears {count} times")
            
            print()
            print("🔧 FIX: Make sure each GOOGLE_API_KEY_X is different")
            return 1
        else:
            print()
            print(f"✅ ALL {num_total} KEYS ARE UNIQUE!")
            print("   No duplicates detected - configuration is correct!")
        
        print()
        print("-"*70)
        print()
        
        # Performance estimates
        if num_total == 1:
            print("⚡ Performance Estimate:")
            print(f"   With 1 key: ~10 requests/minute")
            print()
            print("💡 TIP: Add more keys for faster experiments!")
            print("   With 5 keys: ~50 requests/minute (5x faster)")
        elif num_total < 5:
            print("⚡ Performance Estimate:")
            print(f"   With {num_total} keys: ~{num_total * 10} requests/minute")
            print()
            print(f"💡 TIP: Add {5 - num_total} more key(s) to reach 50 req/min")
        else:
            print("⚡ Performance Estimate:")
            print(f"   With {num_total} keys: ~{num_total * 10} requests/minute")
            print()
            print("🎉 EXCELLENT! You can run experiments at full speed!")
        
        print()
        print("="*70)
        print("✅ CONFIGURATION VERIFIED - READY TO GO!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

