
from tqdm import tqdm
import multiprocessing

def apply_function(func_to_apply, queue_in, queue_out):
    while not queue_in.empty():
        num, obj = queue_in.get()
        queue_out.put((num, func_to_apply(obj)))

def p_map(func, items, cpus=None, verbose=True):
    if cpus is None: cpus = min(multiprocessing.cpu_count(), 32)
    q_in  = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    new_proc  = lambda t,a: multiprocessing.Process(target=t, args=a)
    processes = [new_proc(apply_function, (func, q_in, q_out)) for x in range(cpus)]
    sent = [q_in.put((i, x)) for i, x in enumerate(items)]
    for proc in processes:
        proc.daemon = True
        proc.start()
    if verbose: results = [q_out.get() for x in tqdm(range(len(sent)))]
    else: results = [q_out.get() for x in range(len(sent))]
    for proc in processes: proc.join()
    return [x for i, x in sorted(results)]

