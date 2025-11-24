import multiprocessing
import queue
import time
import os
from stt_handler import run_stt_process

class STTProcessManager:
    def __init__(self):
        self.audio_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.command_queue = multiprocessing.Queue()
        self.process = None

    def start(self):
        """Starts the STT process if it's not running."""
        if self.process and self.process.is_alive():
            print("STT Process is already running.")
            return

        print("Starting STT Process...")
        self.process = multiprocessing.Process(
            target=run_stt_process, 
            args=(self.audio_queue, self.result_queue, self.command_queue)
        )
        self.process.daemon = True
        self.process.start()
        print(f"STT Process started with PID: {self.process.pid}")

    def stop(self):
        """Gracefully stops the STT process."""
        if not self.process:
            return

        print("Stopping STT Process...")
        
        # 1. Send exit signal (optional, if supported by handler)
        # self.command_queue.put(("EXIT", None))

        # 2. Drain queues to prevent deadlock
        self._drain_queues()

        # 3. Terminate
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            
            if self.process.is_alive():
                print("STT Process did not exit in time. Killing...")
                self.process.kill()
                self.process.join()
        
        print("STT Process stopped.")
        self.process = None

    def _drain_queues(self):
        """Empty queues to prevent deadlock during join."""
        print("Draining queues...")
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
            
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                
            while not self.command_queue.empty():
                self.command_queue.get_nowait()
        except Exception as e:
            print(f"Error draining queues: {e}")

    def restart(self):
        """Restarts the STT process."""
        self.stop()
        self.start()

    def is_alive(self):
        return self.process is not None and self.process.is_alive()
