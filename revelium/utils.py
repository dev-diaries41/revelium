
import tracemalloc
import time

def with_mem_profile(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024**2:.2f} MB | Peak memory usage: {peak / 1024**2:.2f} MB")
        return result
    return wrapper


def with_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f" Time elapsed: {time_elapsed:.3f} s")
        return result
    return wrapper

