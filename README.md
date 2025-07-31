# Grant Outreach Generator - Cross-Platform AI Outreach Automation

Automatically generate highly personalized LinkedIn outreach messages using multiple AI providers with real-time web search capabilities. Runs on Windows, macOS, and Linux.

## 🚀 Features

- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Multi-LLM Rotation**: Uses 5 AI providers (Gemini, OpenAI, Groq, DeepSeek, Qwen)
- **Auto Web Search**: Real-time company research via Google Search
- **Smart Rate Limiting**: Automatic provider switching and retry logic
- **Resume Processing**: Never lose progress - continues where it left off
- **Graceful Shutdown**: Proper Ctrl+C handling with progress saving
- **Auto Virtual Environment**: Automatically creates and activates Python environment

## 📋 Quick Start

### First Time Setup

```bash
# Clone repository (if not already done)
git clone <repository-url>
cd sleftshreds-personalized-outreach-v0

# One-command setup and start
python outreach_controller.py --setup
```

This automatically:
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies
- ✅ Prompts for API keys (OpenAI, Gemini, Groq, OpenRouter)
- ✅ Sets up directories and logging

### Start Processing

```bash
# Start daemon
python outreach_controller.py start

# Start and resume existing progress
python outreach_controller.py start --resume

# Test mode (process only 3 people)
python outreach_controller.py test --run_count 3
```

### Drop Files & Monitor

1. **📁 Input**: Drop LinkedIn CSV files in `./datasets/`
2. **📊 Output**: Check `./outputs/` for `[filename]_outreach.csv`
3. **📋 Monitor**: 
   ```bash
   # Windows
   Get-Content logs/outreach.log -Wait
   
   # macOS/Linux
   tail -f logs/outreach.log
   ```

### Stop Processing

```bash
# Stop daemon
python outreach_controller.py stop

# Force stop with Ctrl+C (works properly now!)
# Press Ctrl+C in terminal - stops within 1-2 seconds
```

## 🤖 AI Providers Supported

The system automatically rotates between these providers:

| Provider | Model | Web Search | Cost/1K | Notes |
|----------|-------|------------|---------|--------|
| **Gemini** | gemini-2.0-flash | ✅ Google Search | $0.002 | **Best quality** |
| **OpenAI** | gpt-4o | ❌ | $0.005 | High quality |
| **Groq** | llama-3.3-70b | ❌ | $0.001 | Fast & cheap |
| **DeepSeek** | deepseek-chat-v3 | ❌ | Free | Free tier |
| **Qwen** | qwen3-coder | ❌ | Free | Free tier |

## 🔧 Platform-Specific Commands

### Windows
```cmd
# Setup
python outreach_controller.py --setup

# Start
python outreach_controller.py start

# Monitor
Get-Content logs/outreach.log -Wait

# Stop
python outreach_controller.py stop
```

### macOS/Linux
```bash
# Setup  
python3 outreach_controller.py --setup

# Start
python3 outreach_controller.py start  

# Monitor
tail -f logs/outreach.log

# Stop
python3 outreach_controller.py stop
```

## 📁 Directory Structure

```
sleftshreds-personalized-outreach-v0/
├── datasets/           # 📥 Drop LinkedIn CSV files here
├── outputs/           # 📊 Generated outreach files
├── logs/              # 📋 System logs and state
├── .venv/             # 🐍 Auto-created virtual environment
├── .env               # 🔑 API keys (auto-created)
├── outreach_controller.py  # 🎛️ Main control script
├── outreach_daemon.py      # 🤖 Processing engine
└── README.md          # 📖 This file
```

## 🔑 API Keys Setup

During setup, you'll be prompted for:

1. **OpenAI API Key** - Get from [platform.openai.com](https://platform.openai.com/api-keys)
2. **Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Groq API Key** - Get from [Groq Console](https://console.groq.com/)
4. **OpenRouter API Key** - Get from [OpenRouter](https://openrouter.ai/keys)

> **Note**: You need at least 2 API keys for the rotation system to work properly.

## 📊 Output Format

Generated CSV files contain:

| Column | Description |
|--------|-------------|
| `firstName` | Person's first name |
| `lastName` | Person's last name |
| `email` | Email address |
| `linkedinUrl` | LinkedIn profile URL |
| `companyName` | Company name |
| `jobTitle` | Job title |
| `emailSubject` | AI-generated subject line |
| `emailBody` | Personalized outreach message |
| `citations` | Web sources used (Gemini only) |

## 🛠️ Commands Reference

### Main Commands
```bash
# Setup
python outreach_controller.py --setup

# Start processing
python outreach_controller.py start

# Start with resume
python outreach_controller.py start --resume

# Test mode
python outreach_controller.py test --run_count 5

# Check status
python outreach_controller.py status

# Stop daemon
python outreach_controller.py stop

# Restart
python outreach_controller.py restart
```

### Management Commands
```bash
# Clean logs and outputs
python outreach_controller.py restart  # (without --resume)

# View status
python outreach_controller.py status

# Monitor live
tail -f logs/outreach.log   # macOS/Linux
Get-Content logs/outreach.log -Wait  # Windows
```

## 🚨 Rate Limiting & Error Handling

### Automatic Rate Limit Management
- ✅ **Per-Provider Limits**: Each AI provider has independent rate limits
- ✅ **Auto Switching**: Automatically switches to available providers
- ✅ **Smart Retry**: Waits appropriate time before retrying rate-limited providers
- ✅ **Progress Saving**: Never loses progress during rate limits

### What You'll See
```
⏳ gemini rate limited until 2025-01-31 15:30:00
🤖 Using openai (gpt-4o)
✅ Successfully generated email for John Smith
```

## 🐛 Troubleshooting

### Common Issues

**"No virtual environment found"**
```bash
python outreach_controller.py --setup
```

**"No API keys found"**
```bash
python outreach_controller.py --setup
# Enter at least 2 API keys
```

**Ctrl+C doesn't stop (OLD ISSUE - NOW FIXED!)**
- ✅ **Fixed**: Now stops within 1-2 seconds
- ✅ **Saves Progress**: Automatically saves current progress
- ✅ **Clean Exit**: Properly cleans up processes

**Rate limit errors**
```
# Check which providers are rate limited
python outreach_controller.py status

# Wait or add more API keys
python outreach_controller.py --setup
```

**Process seems stuck**
```bash
# Check logs for current status
tail -20 logs/outreach.log

# Check if actually processing
python outreach_controller.py status
```

### Platform-Specific Issues

**Windows: "python not found"**
```cmd
# Use python instead of python3
python outreach_controller.py --setup
```

**macOS: "command not found: python3"**
```bash
# Install Python via Homebrew
brew install python
```

**Linux: "python3-venv not found"**
```bash
# Ubuntu/Debian
sudo apt install python3-venv python3-pip

# CentOS/RHEL
sudo yum install python3-venv python3-pip
```

## 💡 Tips & Best Practices

### For Best Results
1. **Use Multiple API Keys**: Set up 3-4 providers for best reliability
2. **Monitor Progress**: Check `outputs/` folder periodically
3. **Resume Feature**: Always use `--resume` when restarting
4. **Rate Limits**: Be patient - high-quality generation takes time

### Performance Optimization
- **Gemini First**: Provides best results with web search
- **Automatic Switching**: System optimizes provider selection
- **Batch Processing**: Processes files sequentially for stability

### Cost Management
- **Free Tiers**: DeepSeek and Qwen are free
- **Gemini**: Best quality at $0.002/request
- **Total Cost**: ~$2-5 per 1000 profiles processed

## 🔄 Development & Updates

### Current Version Features
- ✅ Cross-platform compatibility (Windows/macOS/Linux)
- ✅ 5 LLM providers with automatic rotation
- ✅ Fixed infinite loop on Ctrl+C
- ✅ Enhanced error handling and logging
- ✅ Auto virtual environment management
- ✅ Resume functionality with progress tracking
- ✅ Graceful shutdown with progress saving

### System Requirements
- **Python**: 3.6+ (3.8+ recommended)
- **RAM**: 512MB+ available
- **Storage**: 100MB+ for dependencies
- **Network**: Internet connection for AI APIs
- **Platforms**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 📞 Support

### Log Analysis
```bash
# View recent errors
grep "ERROR" logs/outreach.log | tail -10

# View processing progress  
grep "Processing:" logs/outreach.log | tail -10

# View rate limit status
grep "rate limited" logs/outreach.log | tail -5
```

### Getting Help
1. **Check Status**: `python outreach_controller.py status`
2. **View Logs**: `tail -f logs/outreach.log`
3. **Restart Fresh**: `python outreach_controller.py restart`
4. **Re-setup**: `python outreach_controller.py --setup`

---

**🎉 Ready to generate personalized outreach at scale!**

Drop your LinkedIn CSV files in `./datasets/` and watch the magic happen in `./outputs/`!
