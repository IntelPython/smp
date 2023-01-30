# Static Multi-Processing
**SMP** module allows to set static affinity mask for each process inside process pool to limit total
number of threads running in application:
```
python -m smp [-f <oversubscription_factor>] [-p <number_of_outermost_processes>] script.py
```
The module supports two types of process pool: multiprocessing.pool.Pool and
concurrent.futures.ProcessPoolExecutor, as well as one thread pool: multiprocessing.pool.ThreadPool.
Can be run with TBB module as well:
```
python -m smp [-f <oversubscription_factor>] [-p <number_of_outermost_processes>] -m tbb script.py
```
