# 🔧 Gemini API Errors - Complete Guide

This guide explains common errors you might encounter with the Gemini API and how to fix them.

---

## 📌 Table of Contents

1. [finish_reason=2 (MAX_TOKENS)](#finish_reason2-max_tokens)
2. [finish_reason=3 (SAFETY)](#finish_reason3-safety)
3. [finish_reason=4 (RECITATION)](#finish_reason4-recitation)
4. [Rate Limit / Quota Exceeded](#rate-limit--quota-exceeded)
5. [API Key Not Found](#api-key-not-found)

---

## finish_reason=2 (MAX_TOKENS)

### Error Message
```
ERROR: Invalid operation: The `response.text` quick accessor requires the response 
to contain a valid `Part`, but none were returned. The candidate's [finish_reason] is 2.
```

### What It Means
The model hit the **maximum output token limit** before completing its response. When this happens:
- Gemini stops generating mid-sentence
- No text is returned (empty response)
- `response.text` throws an error

### Root Cause
Your `max_output_tokens` setting is **too low** for the complexity of the problem. Math problems with detailed step-by-step solutions often need 800-1500 tokens.

### ✅ Solutions

#### Solution 1: Increase max_tokens (Recommended)
```bash
# Run with higher token limit
python experiments/run_comparison.py --samples 10 --max-tokens 1024

# For very complex problems
python experiments/run_comparison.py --samples 10 --max-tokens 2048
```

The default has been updated from 512 → **1024 tokens**.

#### Solution 2: Check Your Configuration
```python
# In your code, increase the generation config
generation_config=genai.types.GenerationConfig(
    max_output_tokens=1024,  # Increased from 512
    temperature=0
)
```

#### Solution 3: Verify Fix is Applied
After updating the code (as of this guide), the error is now handled gracefully:
```python
# Old behavior: Crashes with error
# New behavior: Shows clear message
prediction = "[INCOMPLETE - Hit max tokens]"
```

### 📊 Recommended Token Limits

| Problem Complexity | Recommended `max_tokens` |
|-------------------|-------------------------|
| Simple arithmetic (5+3) | 256 |
| Basic word problems | 512 |
| **GSM8K dataset** | **1024** ⭐ (default) |
| Complex multi-step | 1536-2048 |

### 🔍 How to Detect
Look for these patterns in your output:
```
[2/5] Processing... 🔑 Using API key #3/5
ERROR: Invalid operation: The `response.text` quick accessor...
✗ | 3.32s | 0→0 tokens  ← Zero tokens means incomplete response
```

The `0→0 tokens` is the key indicator.

---

## finish_reason=3 (SAFETY)

### What It Means
Content was **blocked by Gemini's safety filters**. The API thinks the content might be:
- Harmful or dangerous
- Sexually explicit
- Hate speech
- Violence-related

### Why This Happens with Math Problems
Rarely happens with GSM8K, but can occur if:
- Problem description contains sensitive words (names, locations)
- Generated text accidentally matches a filter pattern

### ✅ Solution
```python
# Add safety settings to your model configuration
from google.generativeai.types import HarmCategory, HarmBlockThreshold

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    tools=[slm_help_tool],
    system_instruction=INSTRUCTIONS,
    safety_settings=safety_settings  # ← Add this
)
```

**⚠️ Warning:** Only disable safety settings if you're confident in your input data (like GSM8K, which is safe math problems).

---

## finish_reason=4 (RECITATION)

### What It Means
The model detected that it was about to **copy content verbatim** from its training data. Google blocks this to prevent copyright issues.

### Why This Happens
- Model recognizes the exact problem from training
- Generated solution matches training data too closely

### ✅ Solution
This is rare and usually okay to fail. The model is being cautious. You can:

1. **Rephrase the prompt** slightly
2. **Add variation** to the problem statement
3. **Accept the failure** (it's doing the right thing)

No configuration change needed.

---

## Rate Limit / Quota Exceeded

### Error Message
```
ResourceExhausted: 429 Quota exceeded for quota metric 'Generate Content API requests per minute'
```

### What It Means
You've hit the **free tier rate limit**:
- **10 requests per minute** per API key
- **1500 requests per day** per API key

### ✅ Solutions

#### Solution 1: Use Multiple API Keys (Recommended)
Set up 5 API keys for **5x speed** (50 requests/minute):

```bash
# On Kaggle: Add secrets
GOOGLE_API_KEY_1 = your-first-key
GOOGLE_API_KEY_2 = your-second-key
GOOGLE_API_KEY_3 = your-third-key
GOOGLE_API_KEY_4 = your-fourth-key
GOOGLE_API_KEY_5 = your-fifth-key

# Verify they're all different
!python tools/check_api_keys.py
```

See [KAGGLE_API_KEYS_SETUP.md](KAGGLE_API_KEYS_SETUP.md) for detailed setup.

#### Solution 2: Reduce Request Rate
```bash
# Test with fewer samples
python experiments/run_comparison.py --samples 5

# Skip SLM baseline (faster)
python experiments/run_comparison.py --samples 10 --skip-slm
```

#### Solution 3: Wait and Retry
The limit resets after 60 seconds. The API key manager handles this automatically.

---

## API Key Not Found

### Error Message
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

### ✅ Solutions

#### Check 1: Verify API Key Exists
```bash
python tools/check_api_keys.py
```

Expected output:
```
✅ FOUND 5 API KEY(S) from Kaggle Secrets
```

#### Check 2: Verify API Key is Valid
1. Go to https://aistudio.google.com/app/apikey
2. Check if your key is still active
3. Generate a new one if needed

#### Check 3: Check Environment Variables
**On Kaggle:**
```python
# Make sure secrets are added in notebook settings
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
key = secrets.get_secret("GOOGLE_API_KEY_1")
print(f"Key loaded: {key[:10]}..." if key else "❌ Not found")
```

**Locally:**
```bash
# Check .env file
cat .env | grep GOOGLE_API_KEY

# Or check environment
echo $GOOGLE_API_KEY_1
```

---

## 🔍 Debugging Tools

### Quick API Key Check
```bash
python tools/check_api_keys.py
```

Shows:
- ✅ How many keys detected
- ✅ Preview of each key (masked)
- ✅ Duplicate detection
- ✅ Performance estimates

### Full Environment Check
```bash
python tools/check_setup.py
```

Verifies:
- Python version
- Dependencies
- GPU availability
- API keys
- Required files
- Model cache

---

## 📚 Quick Reference

### Finish Reason Codes

| Code | Name | Meaning | Action |
|------|------|---------|--------|
| 1 | STOP | ✅ Success | None needed |
| 2 | MAX_TOKENS | ⚠️ Hit token limit | Increase `max_tokens` |
| 3 | SAFETY | 🚫 Safety filter | Adjust safety settings |
| 4 | RECITATION | 🚫 Copyright block | Rephrase prompt |
| 5 | OTHER | ❓ Other issue | Check logs |

### Default Settings (Updated)

```python
max_output_tokens = 1024  # Increased from 512
temperature = 0           # Deterministic
api_keys = 5              # For rate limit handling
cooldown_seconds = 1      # Between API key switches
```

---

## 🆘 Still Having Issues?

1. **Check logs** - Error messages often contain helpful details
2. **Run diagnostics** - `python tools/check_setup.py`
3. **Verify API keys** - `python tools/check_api_keys.py`
4. **Review recent changes** - Check if you modified any configuration
5. **Try with minimal example** - Run `python demo.py` first

---

## 📖 Related Documentation

- [KAGGLE_API_KEYS_SETUP.md](KAGGLE_API_KEYS_SETUP.md) - Multiple API key setup
- [API_KEY_ROTATION_SUMMARY.md](API_KEY_ROTATION_SUMMARY.md) - How key rotation works
- [README.md](README.md) - Full project documentation
- [Google Gemini API Docs](https://ai.google.dev/api/generate-content)

---

**Last Updated:** October 2024  
**Applies to:** Gemini 2.5 Flash via `google-generativeai` SDK

