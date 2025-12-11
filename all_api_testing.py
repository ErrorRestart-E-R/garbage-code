"""
Pre-flight API Testing Module for AI VTuber Bot

This module checks all required APIs, libraries, and services before bot startup.
All tests must pass for the bot to start successfully.
"""

import os
import sys
import importlib.util
from pathlib import Path
import asyncio
from urllib.parse import urlparse


def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print('='*60)


def print_success(message):
    """Print success message"""
    # Windows(cp949) 콘솔에서도 안전하게 출력되도록 ASCII만 사용
    print(f"[OK] {message}")


def print_error(message):
    """Print error message"""
    print(f"[FAIL] {message}")


def print_info(message):
    """Print info message"""
    print(f"  - {message}")


def test_environment_variables() -> bool:
    """
    Tests that all required environment variables are set.
    Returns True if test passes, False otherwise.
    """
    print_test_header("Environment Variables")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check Discord token
        discord_token = os.getenv("DISCORD_TOKEN")
        if not discord_token:
            print_error("DISCORD_TOKEN not found in environment variables")
            print_info("Please add DISCORD_TOKEN to your .env file")
            return False
        
        print_success("DISCORD_TOKEN is set")
        print_success("Environment variables test passed")
        return True
        
    except Exception as e:
        print_error(f"Environment variables test failed: {e}")
        return False


def test_required_packages() -> bool:
    """
    Tests that all required Python packages are installed.
    Returns True if test passes, False otherwise.
    """
    print_test_header("Required Python Packages")
    
    required_packages = {
        'openai': 'openai',
        'discord': 'discord.py',
        'discord.ext.voice_recv': 'discord-ext-voice-recv',
        'faster_whisper': 'faster-whisper',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'dotenv': 'python-dotenv',
        'numpy': 'numpy',
        'chromadb': 'chromadb',
        'mem0': 'mem0ai',
        'sentence_transformers': 'sentence-transformers',
        'pyvts': 'pyvts',
    }
    
    all_installed = True
    install_hint = "pip install -r requirements.txt"
    
    for module_name, package_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None

        if spec is not None:
            print_success(f"{package_name} is installed")
            continue

        print_error(f"{package_name} is NOT installed")
        print_info(f"Install with: {install_hint}")
        all_installed = False
    
    if all_installed:
        print_success("All required packages are installed")
        return True
    else:
        print_error("Some required packages are missing")
        return False


def test_llama_cpp_connection_and_model() -> bool:
    """
    Tests llama.cpp server connection and checks if the model is available.
    Returns True if test passes, False otherwise.
    """
    print_test_header("llama.cpp Connection & Model Check")
    
    try:
        from openai import OpenAI
        import config
        
        print_info(f"Connecting to llama.cpp at {config.LLAMA_CPP_BASE_URL}...")
        
        client = OpenAI(
            base_url=config.LLAMA_CPP_BASE_URL,
            api_key=config.LLAMA_CPP_API_KEY
        )
        
        # Check if server is running by listing models
        try:
            models_response = client.models.list()
            models = list(models_response)
            print_success("Connected to llama.cpp server successfully")
        except Exception as e:
            print_error(f"Cannot connect to llama.cpp server: {e}")
            print_info("Make sure llama.cpp server is running")
            return False
            
        # Get model name from config
        model_name = getattr(config, 'LLM_MODEL_NAME', 'model')
        print_info(f"Checking for model: {model_name}")
        
        # Check if model is available (llama.cpp usually has one model)
        if models:
            print_success(f"Model(s) available on server")
            for m in models:
                print_info(f" - {m.id}")
        else:
            print_info("No models listed, but server is responding")
        
        # Test model with a simple prompt
        print_info("Testing model response...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Hello'}],
                max_tokens=50
            )
            content = response.choices[0].message.content
                
            if content:
                print_success(f"Model response received: {content[:50]}...")
                print_success("llama.cpp connection and model test passed")
                return True
            else:
                print_error("Model did not generate a valid response")
                return False
        except Exception as e:
            print_error(f"Model generation failed: {e}")
            return False
            
    except Exception as e:
        print_error(f"llama.cpp test failed: {e}")
        return False


def test_tts_server() -> bool:
    """
    Tests TTS server connectivity.
    Returns True if test passes, False otherwise.
    """
    print_test_header("TTS Server Connection")
    
    try:
        import requests
        import config
        
        tts_url = config.TTS_SERVER_URL
        print_info(f"Testing connection to: {tts_url}")
        
        # Simple ping to check if server is reachable
        response = requests.get(tts_url.replace('/tts', ''), timeout=5)
        
        if response.status_code in [200, 404, 405]:  # Server is responding
            print_success(f"TTS server is reachable at {tts_url}")
            print_success("TTS server connection test passed")
            return True
        else:
            print_error(f"TTS server returned unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print_error("TTS server connection timed out")
        print_info(f"Check if TTS server is running at {config.TTS_SERVER_URL}")
        return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to TTS server")
        print_info(f"Check if TTS server is running at {config.TTS_SERVER_URL}")
        return False
    except Exception as e:
        print_error(f"TTS server test failed: {e}")
        return False


def test_stt_models() -> bool:
    """
    Tests STT model availability and optional CUDA support.
    Returns True if test passes, False otherwise.
    """
    print_test_header("STT/VAD Models and CUDA")
    
    try:
        import config
        
        try:
            import torch  # Optional dependency
        except ImportError:
            torch = None
            print_info("torch not installed - skipping CUDA diagnostics")
        
        # Check CUDA availability only if torch exists
        if torch:
            if torch.cuda.is_available():
                print_success(f"CUDA is available (Device: {torch.cuda.get_device_name(0)})")
            else:
                print_error("CUDA is NOT available")
                print_info("STT will run on CPU (slower)")
        else:
            print_success("Continuing without CUDA diagnostics")
        
        # Check if faster-whisper can be imported
        try:
            from faster_whisper import WhisperModel
            print_success("faster-whisper is available")
        except ImportError:
            print_error("faster-whisper cannot be imported")
            return False
        
        # Check model ID
        print_info(f"STT Model: {config.STT_MODEL_ID}")
        print_info(f"STT Device: {config.STT_DEVICE}")
        print_info(f"STT Compute Type: {config.STT_COMPUTE_TYPE}")
        
        # Note: We don't actually load the models here as they're heavy
        # They will be loaded in the STT process
        print_success("STT configuration is valid")
        print_success("STT models test passed")
        return True
        
    except Exception as e:
        print_error(f"STT models test failed: {e}")
        return False


def test_reference_files() -> bool:
    """
    Tests that required reference files exist.
    Returns True if test passes, False otherwise.
    """
    print_test_header("Reference Files")
    
    try:
        import config
        
        ref_file = Path(config.TTS_REFERENCE_FILE)
        
        if ref_file.exists():
            print_success(f"TTS reference file found: {config.TTS_REFERENCE_FILE}")
            print_info(f"File size: {ref_file.stat().st_size} bytes")
            print_success("Reference files test passed")
            return True
        else:
            print_error(f"TTS reference file not found: {config.TTS_REFERENCE_FILE}")
            print_info("The bot can still work, but TTS quality may not be optimal")
            # Return True anyway since it's not critical
            return True
            
    except Exception as e:
        print_error(f"Reference files test failed: {e}")
        # Not critical, return True
        return True


def _parse_ws_url(ws_url: str) -> tuple[str, int]:
    """
    ws://localhost:8001 형태를 (host, port)로 변환.
    """
    u = urlparse(ws_url)
    host = u.hostname or "localhost"
    port = int(u.port or 8001)
    return host, port


def test_vts_connection_optional() -> bool:
    """
    VTube Studio 연결 스모크 테스트.
    - config.VTS_ENABLED=True일 때만 실행
    - 토큰 발급/인증은 하지 않음(사용자 Allow 팝업 유발 방지)
    """
    print_test_header("VTube Studio (VTS) Connection (optional)")

    try:
        import config

        if not getattr(config, "VTS_ENABLED", False):
            print_info("VTS is disabled in config.py (VTS_ENABLED=False) - skipping")
            return True

        try:
            import pyvts  # type: ignore
        except ImportError:
            print_error("pyvts is NOT installed")
            print_info("Install with: pip install pyvts")
            return False

        host, port = _parse_ws_url(getattr(config, "VTS_WS_URL", "ws://localhost:8001"))
        token_path = getattr(config, "VTS_AUTH_TOKEN_PATH", "./vts_token.txt")
        plugin_name = getattr(config, "VTS_PLUGIN_NAME", "AiVutber")
        plugin_dev = getattr(config, "VTS_PLUGIN_DEVELOPER", "ErrorRestart")

        async def _run():
            plugin_info = {
                "plugin_name": plugin_name,
                "developer": plugin_dev,
                "authentication_token_path": token_path,
            }
            v = pyvts.vts(plugin_info=plugin_info, host=host, port=port)
            await v.connect()
            await v.close()

        asyncio.run(_run())
        print_success(f"Connected to VTS API at ws://{host}:{port}")
        print_success("VTS connection test passed")
        return True

    except Exception as e:
        print_error(f"VTS connection test failed: {e}")
        print_info("Make sure VTube Studio is running and Settings > API > Enable API is ON")
        print_info("Also ensure the API port matches VTS_WS_URL in config.py")
        return False


def run_all_tests() -> bool:
    """
    Runs all pre-flight tests.
    Returns True if all tests pass, False otherwise.
    Note:
    - 이 단계는 "사전 점검"이 목적이며, 무거운 모델(STT 등)을 실제로 로딩/다운로드하지 않습니다.
    - STT(faster-whisper) 모델 로딩은 실제 실행 시 STT 프로세스(`stt_handler.run_stt_process`)에서 수행됩니다.
    """
    import os
    import platform
    
    def clear_terminal():
        """Clear terminal screen"""
        if platform.system() == "Windows":
            os.system('cls')
        else:
            os.system('clear')
    
    # Clear screen at start
    clear_terminal()
    
    print("\n" + "="*60)
    print("AI VTuber Bot - Pre-Flight System Check")
    print("="*60)
    print("\nRunning tests... (will stop on first failure)\n")
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Required Packages", test_required_packages),
        ("llama.cpp Connection & Model Check", test_llama_cpp_connection_and_model),
        ("TTS Server", test_tts_server),
        ("STT/VAD Models", test_stt_models),
        ("Reference Files", test_reference_files),
        ("VTube Studio (optional)", test_vts_connection_optional),
    ]
    
    passed_tests = []
    
    for i, (test_name, test_func) in enumerate(tests, 1):
        try:
            result = test_func()
            
            if result:
                passed_tests.append(test_name)
                # Clear and show all passed tests so far
                clear_terminal()
                print()
                for j, passed_name in enumerate(passed_tests, 1):
                    print(f"[{j}/{len(tests)}] [OK] {passed_name} - PASSED")
                
                if i < len(tests):
                    print("\nContinuing to next test...\n")
                    import time
                    time.sleep(0.5)  # Brief pause to show success
            else:
                print(f"\n\n[FAIL] Test failed: {test_name}")
                print("="*60)
                print("Pre-flight checks stopped due to failure.")
                print("Please fix the error above before continuing.")
                print("="*60 + "\n")
                return False
                
        except Exception as e:
            print(f"\n\n[FAIL] Test '{test_name}' crashed: {e}")
            print("="*60)
            print("Pre-flight checks stopped due to crash.")
            print("="*60 + "\n")
            return False
    
    # All tests passed - clear and show final summary
    clear_terminal()
    print("\n" + "="*60)
    print("ALL PRE-FLIGHT CHECKS PASSED")
    print("="*60)
    print(f"\nCompleted {len(tests)}/{len(tests)} tests successfully:")
    for i, test_name in enumerate(passed_tests, 1):
        print(f"  [{i}/{len(tests)}] [OK] {test_name}")
    print("\n" + "="*60)
    print("Bot is ready to start!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    # Allow running this module standalone for testing
    success = run_all_tests()
    sys.exit(0 if success else 1)
