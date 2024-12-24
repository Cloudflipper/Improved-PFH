import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.clip_time = None

    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None
        self.clip_time = time.time()
        print("Timer started!")

    def stop(self):
        """Stop the timer and calculate the elapsed time"""
        if self.start_time is None:
            print("Timer has not been started yet!")
            return
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Timer stopped! Elapsed time: {self.elapsed_time:.6f} seconds.")

    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.clip_time = None
        print("Timer reset!")

    def get_elapsed_time(self):
        """Get the recorded elapsed time"""
        if self.elapsed_time is None:
            print("No elapsed time recorded!")
            return None
        return self.elapsed_time

    def clip(self):
        """Calculate the time elapsed since the last clip and update the clip time"""
        cnt = time.time() - self.clip_time
        self.clip_time = time.time()
        return cnt
