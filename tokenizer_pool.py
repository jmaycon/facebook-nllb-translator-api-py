import queue
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TokenizerPool:
    def __init__(self, model_name: str, initial_capacity: int = 50, min_capacity: int = 50,
                 max_capacity: int = 150, idle_timeout: float = 300.0):
        self.tokenizer_class = AutoTokenizer
        self.model_name = model_name
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.idle_timeout = idle_timeout

        self.pool = queue.Queue(maxsize=max_capacity)
        self.total_created = 0
        self.lock = threading.Lock()
        self.creation_lock = threading.Lock()

        logger.info(f"Initializing TokenizerPool with {initial_capacity} tokenizers...")

        with ThreadPoolExecutor(max_workers=min(initial_capacity, 20)) as executor:
            futures = [
                executor.submit(self._create_and_put_tokenizer, i + 1, initial_capacity)
                for i in range(initial_capacity)
            ]
            for future in futures:
                future.result()

        logger.info("TokenizerPool initialized")

        threading.Thread(target=self._clean_idle_tokenizers, daemon=True).start()

    def _create_and_put_tokenizer(self, index: int, total: int):
        tokenizer = self._create_tokenizer(index, total)
        self.pool.put((time.time(), tokenizer))
        with self.lock:
            self.total_created += 1

    def _create_tokenizer(self, index=None, total=None):
        if index and total:
            logger.info(f"Creating tokenizer {index} of {total}")
        else:
            logger.info("Creating new tokenizer")
        return self.tokenizer_class.from_pretrained(self.model_name)

    def acquire(self):
        try:
            _, tokenizer = self.pool.get_nowait()
            logger.debug("Reusing tokenizer from pool")
        except queue.Empty:
            with self.lock:
                if self.total_created < self.max_capacity:
                    self.total_created += 1
                    logger.info(f"Pool exhausted — creating tokenizer (total now: {self.total_created})")
                    return self._create_tokenizer()
            logger.info("Waiting for tokenizer to become available...")
            _, tokenizer = self.pool.get()
        return tokenizer

    def release(self, tokenizer):
        try:
            self.pool.put_nowait((time.time(), tokenizer))
        except queue.Full:
            logger.warning("Tokenizer discarded: pool is full")

    def _clean_idle_tokenizers(self):
        while True:
            time.sleep(self.idle_timeout / 2)
            now = time.time()
            retained = queue.Queue(maxsize=self.max_capacity)
            retained_count = 0
            cleaned = 0

            while not self.pool.empty():
                timestamp, tokenizer = self.pool.get()
                age = now - timestamp
                if age < self.idle_timeout or retained_count < self.min_capacity:
                    retained.put((timestamp, tokenizer))
                    retained_count += 1
                else:
                    with self.lock:
                        self.total_created -= 1
                    cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} idle tokenizer(s) from pool")
            self.pool = retained
