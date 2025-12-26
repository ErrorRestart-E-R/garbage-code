import warnings
import time
import queue
import numpy as np
from logger import setup_logger
import os
import config
from typing import Optional
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Reduce noisy library warnings in STT subprocess console
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

def run_stt_process(audio_queue, result_queue, command_queue, status_queue=None):
    """
    Standalone process for Speech-to-Text.
    Handles lightweight audio buffering and transcription without VAD.
    Running this isolated prevents the Discord bot from freezing during heavy inference.
    """
    print(f"STT Process started. PID: {os.getpid()}")
    
    # --- Model Initialization ---
    # Models must be loaded within this process to avoid CUDA context issues with multiprocessing.
    print("Loading Faster-Whisper model...")

    def _notify(msg):
        if status_queue is None:
            return
        try:
            status_queue.put(msg)
        except Exception:
            pass

    model = None
    device_used = "unknown"
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            config.STT_MODEL_ID,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )
        device_used = str(config.STT_DEVICE)
        print(f"Faster-Whisper loaded ({device_used}).")
    except Exception as e:
        print(f"Failed to load STT model on {config.STT_DEVICE}: {e}. Fallback to CPU base.")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            device_used = "cpu"
            print("Faster-Whisper loaded (cpu fallback).")
        except Exception as e2:
            print(f"Failed to load STT model fallback: {e2}")
            _notify({"type": "STT_ERROR", "error": str(e2)})
            return

    # --- State Management ---
    user_buffers = {}       # Incoming raw audio stream
    user_speech_buffers = {} # Accumulated speech segments
    user_last_activity = {}  # Timestamp for cleanup

    print("STT Process Ready.")
    _notify({"type": "STT_READY", "device": device_used})

    while True:
        # 1. Check Commands
        # NOTE: multiprocessing.Queue.empty()는 경쟁 조건이 있어 신뢰하기 어렵습니다.
        # get_nowait()를 Empty 예외로 drain 하는 방식이 더 안전합니다.
        try:
            while True:
                cmd, data = command_queue.get_nowait()
                logger.debug("Command received")
                if cmd == "LEAVE":
                    user_id = data
                    if user_id in user_buffers:
                        del user_buffers[user_id]
                    if user_id in user_speech_buffers:
                        del user_speech_buffers[user_id]
                    if user_id in user_last_activity:
                        del user_last_activity[user_id]
                    print(f"Cleaned up user {user_id}")
        except queue.Empty:
            pass
        except Exception:
            pass

        # 2. Process Audio
        try:
            # Non-blocking get with timeout to allow cleanup loop to run
            user_id, pcm_data = audio_queue.get(timeout=0.1)
            
            # Update Activity
            user_last_activity[user_id] = time.time()
            
            # Initialize User State
            if user_id not in user_buffers:
                user_buffers[user_id] = bytearray()
                user_speech_buffers[user_id] = bytearray()
            
            user_buffers[user_id].extend(pcm_data)
            
            # Process audio in chunks of 512 samples (32ms)
            while len(user_buffers[user_id]) >= config.FRAME_SIZE_BYTES:
                frame = user_buffers[user_id][:config.FRAME_SIZE_BYTES]
                del user_buffers[user_id][:config.FRAME_SIZE_BYTES]
                user_speech_buffers[user_id].extend(frame)
        except queue.Empty:
            for user_id in list(user_last_activity.keys()):
                if time.time() - user_last_activity[user_id] > config.MIN_SILENCE_DURATION_MS/1000:
                    # Silence detected.
                    if len(user_speech_buffers[user_id]) > 0:
                        # End of speech segment -> Transcribe
                        segment_end_ts = time.time()
                        audio_to_transcribe = user_speech_buffers[user_id][:]
                        user_speech_buffers[user_id] = bytearray()
                        transcribe_and_send(model, user_id, audio_to_transcribe, result_queue, segment_end_ts=segment_end_ts)
        except Exception as e:
            print(f"Error in STT loop: {str(e)}")

        # 3. Cleanup Inactive Users
        current_time = time.time()
        users_to_remove = []
        for uid, last_time in user_last_activity.items():
            if current_time - last_time > config.USER_TIMEOUT_SECONDS:
                users_to_remove.append(uid)
        
        for uid in users_to_remove:
            print(f"User {uid} timed out. Cleaning up.")
            del user_buffers[uid]
            del user_speech_buffers[uid]
            del user_last_activity[uid]

def transcribe_and_send(model, user_id, audio_data, result_queue, segment_end_ts: Optional[float] = None):
    """Enhanced transcription with accuracy optimizations for noisy Discord audio."""
    if len(audio_data) < config.STT_MIN_AUDIO_LENGTH:  # Ignore very short audio
        return

    data_s16 = np.frombuffer(audio_data, dtype=np.int16)
    data_f32 = data_s16.astype(np.float32) / 32768.0

    # Duration gate (prevents tiny clicks/breaths from triggering hallucinated text)
    try:
        duration_seconds = float(len(audio_data)) / (16000.0 * 2.0)  # 16kHz * 16-bit mono
        min_sec = float(getattr(config, "STT_MIN_AUDIO_SECONDS", 0.0) or 0.0)
        if min_sec > 0.0 and duration_seconds < min_sec:
            logger.debug(f"Audio too short ({duration_seconds:.3f}s < {min_sec:.3f}s), skipping")
            return
    except Exception:
        pass
    
    # Check audio energy (RMS) - filter out noise/silence
    rms = np.sqrt(np.mean(data_f32 ** 2))
    if rms < config.STT_MIN_RMS_THRESHOLD:  # Too quiet, likely noise or silence
        logger.debug(f"Audio too quiet (RMS={rms:.4f}), skipping")
        return
    
    # Normalize audio to -0.95 dBFS (RealtimeSTT style)
    if config.STT_NORMALIZE_AUDIO:
        peak = np.max(np.abs(data_f32))
        if peak > 0:
            data_f32 = (data_f32 / peak) * 0.95
    
    # Measure STT inference time precisely
    start_time = time.perf_counter()
    start_time_wall = time.time()
    try:
        # Enhanced transcription with all accuracy parameters
        segments, info = model.transcribe(
            data_f32,
            language=config.STT_LANGUAGE,
            
            # Accuracy parameters
            beam_size=config.STT_BEAM_SIZE,
            best_of=config.STT_BEST_OF,
            patience=config.STT_PATIENCE,

            # Suppress specific tokens
            suppress_tokens=config.STT_SUPPRESS_TOKENS,
            
            # Temperature fallback for difficult audio
            temperature=config.STT_TEMPERATURE,
            
            # Quality thresholds
            compression_ratio_threshold=config.STT_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=config.STT_LOG_PROB_THRESHOLD,
            no_speech_threshold=config.STT_NO_SPEECH_THRESHOLD,
            
            # VAD filtering
            vad_filter=config.STT_VAD_FILTER,
            vad_parameters={
                "threshold": config.STT_VAD_THRESHOLD,
                "min_speech_duration_ms": config.STT_VAD_MIN_SPEECH_MS,
                "min_silence_duration_ms": config.STT_VAD_MIN_SILENCE_MS,
                "speech_pad_ms": config.STT_VAD_SPEECH_PAD_MS,
            },
            
            # Additional options
            condition_on_previous_text=bool(getattr(config, "STT_CONDITION_ON_PREVIOUS_TEXT", False)),
            initial_prompt=None,  # Can add domain-specific prompt here
        )
        
        text = ""
        no_speech_prob_sum = 0
        segment_count = 0
        logprob_sum = 0.0
        logprob_count = 0
        no_speech_margin = float(getattr(config, "STT_POST_FILTER_NO_SPEECH_MARGIN", 0.2))
        
        for segment in segments:
            # Check no_speech_probability - high value means likely hallucination
            if segment.no_speech_prob > float(config.STT_NO_SPEECH_THRESHOLD) + no_speech_margin:
                logger.debug(f"Skipping segment with high no_speech_prob: {segment.no_speech_prob:.2f}")
                continue
            text += segment.text
            no_speech_prob_sum += segment.no_speech_prob
            segment_count += 1

            lp = getattr(segment, "avg_logprob", None)
            if isinstance(lp, (int, float)):
                logprob_sum += float(lp)
                logprob_count += 1
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        text = text.strip()
        
        if text:
            avg_no_speech = no_speech_prob_sum / max(segment_count, 1)
            if logprob_count > 0:
                avg_lp = logprob_sum / max(logprob_count, 1)
                min_lp = float(getattr(config, "STT_POST_FILTER_MIN_AVG_LOGPROB", -999.0))
                if avg_lp < min_lp:
                    logger.debug(
                        f"Skipping low-confidence transcription (avg_logprob={avg_lp:.2f} < {min_lp:.2f}): {text}"
                    )
                    return
            logger.debug(f"Transcription successful for user {user_id}")
            logger.debug(f"Transcription: {text} (avg no_speech_prob: {avg_no_speech:.2f}, latency: {duration:.2f}s)")
            queue_delay_s = None
            if isinstance(segment_end_ts, (int, float)) and segment_end_ts > 0:
                queue_delay_s = float(max(0.0, start_time_wall - float(segment_end_ts)))
            result_queue.put({
                "user_id": user_id,
                "text": text,
                # Backward compatible string
                "latency": f"{duration:.3f}s",
                # Numeric metrics for pipeline timing
                "stt_latency_s": float(duration),
                "stt_audio_s": float(duration_seconds),
                "stt_queue_delay_s": float(queue_delay_s) if queue_delay_s is not None else None,
                "stt_rms": float(rms),
            })
    except Exception as e:
        print(f"Transcription error: {e}")

