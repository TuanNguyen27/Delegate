# tools/check_setup.py
"""
Setup checker for GSM8K comparison experiments
Verifies all dependencies and files are ready
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Python 3.8+ recommended")
        return False
    return True


def check_imports():
    """Check required packages"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'google.generativeai': 'Google Generative AI',
        'pandas': 'Pandas',
        'datasets': 'HuggingFace Datasets',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No CUDA GPU - SLM will run on CPU (slower)")
            return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_env():
    """Check environment variables"""
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env file found")
        
        # Check for Google API key
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('GOOGLE_API_KEY'):
            print("✓ GOOGLE_API_KEY set")
            return True
        else:
            print("✗ GOOGLE_API_KEY not found in .env")
            return False
    else:
        print("✗ .env file not found")
        print("  Create .env with: GOOGLE_API_KEY=your_key_here")
        return False


def check_api_keys():
    """Check and verify multiple API keys"""
    print("\n🔍 Checking API key configuration...")
    
    try:
        # Import the key loading functions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools.api_key_manager import load_api_keys_from_kaggle, load_api_keys_from_env
        
        # Try loading from Kaggle first
        keys = load_api_keys_from_kaggle()
        source = "Kaggle Secrets"
        
        # Fall back to environment
        if not keys:
            from dotenv import load_dotenv
            load_dotenv()
            keys = load_api_keys_from_env()
            source = "Environment Variables"
        
        if not keys:
            print("✗ No API keys found")
            print("  Add keys as:")
            print("    - Kaggle: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")
            print("    - Local: .env file or environment variables")
            return False
        
        print(f"✓ Found {len(keys)} API key(s) from {source}")
        print()
        
        # Show each key with preview
        for i, key in enumerate(keys, 1):
            if len(key) >= 14:
                preview = f"{key[:10]}...{key[-4:]}"
            else:
                preview = f"{key[:6]}...{key[-2:]}"
            print(f"  Key #{i}: {preview}")
        
        # Check for duplicates
        print()
        unique_keys = set(keys)
        if len(unique_keys) < len(keys):
            duplicates = len(keys) - len(unique_keys)
            print(f"⚠️  WARNING: {duplicates} duplicate key(s) detected!")
            print("  Some keys are the same. Each should be unique.")
            
            # Show which keys are duplicates
            from collections import Counter
            key_counts = Counter(keys)
            for key, count in key_counts.items():
                if count > 1:
                    preview = f"{key[:10]}...{key[-4:]}"
                    print(f"    {preview} appears {count} times")
            return False
        else:
            print(f"✓ All {len(keys)} keys are unique (no duplicates)")
        
        # Recommendations
        print()
        if len(keys) == 1:
            print("💡 Tip: Add more keys to avoid rate limits")
            print("   With 1 key: ~10 requests/minute")
            print("   With 5 keys: ~50 requests/minute (5x faster!)")
        elif len(keys) < 5:
            print(f"💡 Tip: You have {len(keys)} keys. Consider adding more!")
            print(f"   Current: ~{len(keys)*10} requests/minute")
            print(f"   With 5 keys: ~50 requests/minute")
        else:
            print(f"🎉 Excellent! {len(keys)} keys = ~{len(keys)*10} requests/minute")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking API keys: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_files():
    """Check required files exist"""
    required_files = [
        'tools/gsm8k_loader.py',
        'experiments/utils.py',
        'experiments/run_comparison.py',
        'experiments/llm_experiment.py',
        'experiments/router_experiment.py',
        'experiments/router_agent.py',
        'experiments/slm_experiment.py',
        'router_agent_demo.py',
        'demo.py'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    return True


def check_model_cache():
    """Check if models are cached"""
    cache_dir = Path.home() / '.cache' / 'huggingface'
    if cache_dir.exists():
        print(f"✓ HuggingFace cache: {cache_dir}")
        
        # Check for Qwen model
        model_files = list(cache_dir.rglob('*Qwen*'))
        if model_files:
            print("✓ Qwen model found in cache")
        else:
            print("⚠️  Qwen model not cached - will download on first run (~3GB)")
    else:
        print("⚠️  No HuggingFace cache - models will download on first run")
    
    return True


def main():
    print("="*70)
    print("GSM8K COMPARISON - SETUP CHECK")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("GPU/CUDA", check_gpu),
        ("Environment Variables", check_env),
        ("API Keys Configuration", check_api_keys),
        ("Required Files", check_files),
        ("Model Cache", check_model_cache)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 70)
        results[name] = check_func()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to run experiments.")
        print("\nQuick start:")
        print("  python run_comparison.py --samples 10 --seed 123")
    else:
        print("\n⚠️  Some checks failed. Fix issues above before running.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())