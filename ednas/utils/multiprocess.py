import os

def get_pool(ptype="thread", num_workers=48):
    from pathos.multiprocessing import ProcessPool, ThreadPool
    if ptype == "thread":
        return ThreadPool(num_workers)
    elif ptype == "process":
        return ProcessPool(num_workers)
    else:
        raise ValueError(ptype)


def get_std_pool(ptype="thread", num_workers=48):
    if ptype == "thread":
        raise Exception("不要用这个函数，会出线程不安全的bug")
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=num_workers)
    elif ptype == "process":
        from multiprocessing import Pool
        return Pool(process=num_workers)
    else:
        raise ValueError(ptype)
