import multiprocessing


def run_parallel(func, configs):
    proc_list = list()
    for config in configs:
        p = multiprocessing.Process(target=func, args=(config,))
        proc_list.append(p)
        p.start()
    # Wait all processes.
    for p in proc_list:
        p.join()


def run_parallel_async(func, configs, pool_size=4):
    pool = multiprocessing.Pool(pool_size)
    results = list()
    for config in configs:
        res = pool.apply_async(func, args=(config,))
        results.append(res)
    pool.close()
    pool.join()
    res = list()
    for item in results:
        res.append(item.get())
    return res
