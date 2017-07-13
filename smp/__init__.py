#!/usr/bin/env python
# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import print_function
import os
import sys
from threading import Lock
from multiprocessing.pool import Pool, ThreadPool
from multiprocessing import cpu_count
try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
except:
    ProcessPoolExecutor = None
    ThreadPoolExecutor = None
from ctypes import *
try:
    from os import sched_getaffinity, sched_setaffinity
except:
    sched_getaffinity = None
    sched_setaffinity = None
try:
    from psutil import Process as PsutilProcess
except:
    PsutilProcess = None

__version__ = "0.1.3"
__all__ = ["Monkey"]
__doc__ = """
Static Multi-Processing module
enables composability of nested parallelism by controlling the number of threads
and setting affinity mask for each Python's worker process or thread, which helps
to limit total number of threads running in application.

Run `python -m smp -h` for command line options.
"""

libc_module_name = "libc.so.6"

oversubscription_factor = 2
max_top_workers = 0

mkl_module_name = "libmkl_rt"
omp_gnu_module_name = "libgomp"
omp_intel_module_name = "libiomp"
found_module_name = None

native_wrapper_list = dict()
native_wrapper_lock = Lock()

class dl_phdr_info32(Structure):
    _fields_ = [("dlpi_addr",  c_uint32),
                ("dlpi_name",  c_char_p),
                ("dlpi_phdr",  c_void_p),
                ("dlpi_phnum", c_uint16)]

class dl_phdr_info64(Structure):
    _fields_ = [("dlpi_addr",  c_uint64),
                ("dlpi_name",  c_char_p),
                ("dlpi_phdr",  c_void_p),
                ("dlpi_phnum", c_uint16)]

def callback(info, size, data):
    global found_module_name
    desired_module = cast(data, c_char_p).value.decode('utf-8')
    is_64bits = sys.maxsize > 2**32
    if is_64bits:
        info64 = cast(info, POINTER(dl_phdr_info64))
        module_name = info64.contents.dlpi_name
    else:
        info32 = cast(info, POINTER(dl_phdr_info32))
        module_name = info32.contents.dlpi_name
    if module_name:
        module_name = module_name.decode("utf-8")
        if module_name.find(desired_module) >= 0:
            found_module_name = module_name
            return 1
    return 0

class NativeWrapper:
    def __init__(self):
        self._load_omp()
        self._load_mkl()

    def is_omp_found(self):
        if self.libomp:
            return True
        return False

    def omp_set_num_threads(self, n):
        if self.libomp:
            try:
                self.libomp.omp_set_num_threads(n)
            except:
                return

    def is_mkl_found(self):
        if self.libmkl:
            return True
        return False

    def mkl_set_num_threads(self, n):
        if self.libmkl:
            try:
                self.libmkl.MKL_Set_Num_Threads(n)
            except:
                return

    def _load_mkl(self):
        try:
            global found_module_name
            self.libc = CDLL(libc_module_name)
            found_module_name = None
            CMPFUNC = CFUNCTYPE(c_int, c_void_p, c_size_t, c_char_p)
            cmp_callback = CMPFUNC(callback)

            data = c_char_p(mkl_module_name.encode('utf-8'))
            res = self.libc.dl_iterate_phdr(cmp_callback, data)
            if res == 1 and found_module_name:
                self.libmkl = CDLL(found_module_name)
            else:
                self.libmkl = None
        except:
            self.libmkl = None

    def _load_omp(self):
        try:
            global found_module_name
            self.libc = CDLL(libc_module_name)
            found_module_name = None
            CMPFUNC = CFUNCTYPE(c_int, c_void_p, c_size_t, c_char_p)
            cmp_callback = CMPFUNC(callback)

            data = c_char_p(omp_gnu_module_name.encode('utf-8'))
            res = self.libc.dl_iterate_phdr(cmp_callback, data)
            if res == 1 and found_module_name:
                self.libomp = CDLL(found_module_name)
            else:
                data = c_char_p(omp_intel_module_name.encode('utf-8'))
                res = self.libc.dl_iterate_phdr(cmp_callback, data)
                if res == 1 and found_module_name:
                    self.libomp = CDLL(found_module_name)
                else:
                    self.libomp = None
        except:
            self.libomp = None

def get_native_wrapper():
    global native_wrapper_list
    global native_wrapper_lock

    native_wrapper_lock.acquire()
    native_wrapper = native_wrapper_list.get(os.getpid())
    if not native_wrapper:
        native_wrapper = NativeWrapper()
        native_wrapper_list[os.getpid()] = native_wrapper
    native_wrapper_lock.release()

    return native_wrapper

def mkl_set_num_threads(n):
    native_wrapper = get_native_wrapper()
    if native_wrapper.is_mkl_found():
        native_wrapper.mkl_set_num_threads(n)

def omp_set_num_threads(n):
    native_wrapper = get_native_wrapper()
    if native_wrapper.is_omp_found():
        native_wrapper.omp_set_num_threads(n)

def get_affinity():
    if sched_getaffinity:
        return sched_getaffinity(0)
    else:
        if PsutilProcess:
            p = PsutilProcess()
            return p.cpu_affinity()
        else:
            return [i for i in range(cpu_count())]

def set_affinity(mask):
    if sched_setaffinity:
        sched_setaffinity(0, mask)
    else:
        if PsutilProcess:
            p = PsutilProcess()
            p.cpu_affinity(mask)
        else:
            if os.name == "posix":
                omp_set_num_threads(len(mask))

def set_proc_affinity(process_count, process_id):
    if process_count == 1:
        return

    cpu_list = list(get_affinity())
    cpu_count = len(cpu_list)

    global oversubscription_factor
    if cpu_count < oversubscription_factor:
        oversubscription_factor = cpu_count

    threads_per_process = oversubscription_factor
    if cpu_count >= process_count:
        threads_per_process = threads_per_process*int(round(float(cpu_count)
                              /float(process_count)))

    start_cpu = (process_id*threads_per_process) % cpu_count;
    mask = [cpu_list[((start_cpu + i) % cpu_count)]
            for i in range(threads_per_process)]
    set_affinity(mask)

    if os.name == "posix":
        mkl_set_num_threads(threads_per_process)

def affinity_worker27(inqueue, outqueue, initializer=None, initargs=(),
                      maxtasks=None, process_count=1, process_id=0):
    from multiprocessing.pool import worker
    set_proc_affinity(process_count, process_id)
    worker(inqueue, outqueue, initializer, initargs, maxtasks)

class AffinityPool27(Pool):
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None):
        if max_top_workers:
            processes = int(max_top_workers)
        Pool.__init__(self, processes, initializer, initargs,
                      maxtasksperchild)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        from multiprocessing.util import debug

        base_id = len(self._pool);
        for i in range(self._processes - len(self._pool)):
            w = self.Process(target=affinity_worker27,
                             args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild,
                                   self._processes, base_id + i)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')

def affinity_worker35(inqueue, outqueue, initializer=None, initargs=(),
                      maxtasks=None, wrap_exception=False,
                      process_count=1, process_id=0):
    from multiprocessing.pool import worker
    set_proc_affinity(process_count, process_id)
    worker(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)

class AffinityPool35(Pool):
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        if max_top_workers:
            processes = int(max_top_workers)
        Pool.__init__(self, processes, initializer, initargs,
                      maxtasksperchild, context)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        from multiprocessing.util import debug

        base_id = len(self._pool);
        for i in range(self._processes - len(self._pool)):
            w = self.Process(target=affinity_worker35,
                             args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild,
                                   self._wrap_exception,
                                   self._processes, base_id + i)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')

def limit_num_threads(process_count, process_id):
    if process_count == 1:
        return

    cpu_list = list(get_affinity())
    cpu_count = len(cpu_list)

    global oversubscription_factor
    if cpu_count < oversubscription_factor:
        oversubscription_factor = cpu_count

    threads_per_process = oversubscription_factor
    if cpu_count >= process_count:
        threads_per_process = threads_per_process*int(round(float(cpu_count)
                              /float(process_count)))

    if os.name == "posix":
        omp_set_num_threads(threads_per_process)
        mkl_set_num_threads(threads_per_process)

def limited_worker27(inqueue, outqueue, initializer=None, initargs=(),
                    maxtasks=None, process_count=1, process_id=0):
    from multiprocessing.pool import worker
    limit_num_threads(process_count, process_id)
    worker(inqueue, outqueue, initializer, initargs, maxtasks)

class LimitedThreadPool27(ThreadPool):
    def __init__(self, processes=None, initializer=None, initargs=()):
        if max_top_workers:
            processes = int(max_top_workers)
        Pool.__init__(self, processes, initializer, initargs)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        from multiprocessing.util import debug

        base_id = len(self._pool)
        for i in range(self._processes - len(self._pool)):
            w = self.Process(target=limited_worker27,
                             args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild,
                                   self._processes, base_id + i)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')

def limited_worker35(inqueue, outqueue, initializer=None, initargs=(),
                    maxtasks=None, wrap_exception=False,
                    process_count=1, process_id=0):
    from multiprocessing.pool import worker
    limit_num_threads(process_count, process_id)
    worker(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)

class LimitedThreadPool35(ThreadPool):
    def __init__(self, processes=None, initializer=None, initargs=()):
        if max_top_workers:
            processes = int(max_top_workers)
        Pool.__init__(self, processes, initializer, initargs)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        from multiprocessing.util import debug

        base_id = len(self._pool);
        for i in range(self._processes - len(self._pool)):
            w = self.Process(target=limited_worker35,
                             args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild,
                                   self._wrap_exception,
                                   self._processes, base_id + i)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')

if ProcessPoolExecutor:

    def affinity_process_worker(call_queue, result_queue,
                                process_count=1, process_id=0):
        from concurrent.futures.process import _process_worker
        set_proc_affinity(process_count, process_id)
        _process_worker(call_queue, result_queue)

    class AffinityProcessPoolExecutor(ProcessPoolExecutor):
        def __init__(self, max_workers=None):
            if max_top_workers:
                max_workers = int(max_top_workers)
            ProcessPoolExecutor.__init__(self, max_workers)

        def _adjust_process_count(self):
            import multiprocessing
            base_id = len(self._processes);
            for i in range(len(self._processes), self._max_workers):
                p = multiprocessing.Process(
                        target=affinity_process_worker,
                        args=(self._call_queue,
                              self._result_queue,
                              self._max_workers, base_id + i))
                p.start()
                self._processes[p.pid] = p

if ThreadPoolExecutor:

    def limited_thread_worker(executor_reference, work_queue,
                              process_count=1, process_id=0):
        from concurrent.futures.thread import _worker
        limit_num_threads(process_count, process_id)
        _worker(executor_reference, work_queue)

    class LimitedThreadPoolExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None):
            if max_top_workers:
                max_workers = int(max_top_workers)
            ThreadPoolExecutor.__init__(self, max_workers)

        def _adjust_thread_count(self):
            import threading, weakref
            from concurrent.futures.thread import _threads_queues
            def weakref_cb(_, q=self._work_queue):
                q.put(None)
            if len(self._threads) < self._max_workers:
                t = threading.Thread(target=limited_thread_worker,
                                     args=(weakref.ref(self, weakref_cb),
                                           self._work_queue,
                                           self._max_workers, 0))
                t.daemon = True
                t.start()
                self._threads.add(t)
                _threads_queues[t] = self._work_queue

class Monkey():
    """
    Context manager which hooks such standard library classes as
    
     Pool, ThreadPool, and ProcessPoolExecutor
    
    It controls number of threads and thread affinity for libraries running
    nested parallel regions. Example:

        with smp.Monkey(oversubscription_factor = 1):
            run_my_parallel_numpy_code()

    """
    _items   = {"Pool"                : None,
                "ThreadPool"          : None,
                "ProcessPoolExecutor" : None}
    _modules = {"Pool"                : None,
                "ThreadPool"          : None,
                "ProcessPoolExecutor" : None}

    def __init__(self, oversubscription_factor=oversubscription_factor, max_top_workers=max_top_workers):
        self.oversubscription_factor = int(oversubscription_factor)
        self.max_top_workers = int(max_top_workers)
        pass

    def _patch(self, class_name, module_name, object):
        self._modules[class_name] = __import__(module_name, globals(),
                                               locals(), [class_name])
        if self._modules[class_name] == None:
            return
        oldattr = getattr(self._modules[class_name], class_name, None)
        if oldattr == None:
            self._modules[class_name] = None
            return
        self._items[class_name] = oldattr
        setattr(self._modules[class_name], class_name, object)

    def __enter__(self):
        global oversubscription_factor, max_top_workers
        oversubscription_factor = self.oversubscription_factor
        max_top_workers = self.max_top_workers
        if sys.version_info.major == 2 and sys.version_info.minor >= 7:
            self._patch("Pool", "multiprocessing.pool", AffinityPool27)
            self._patch("ThreadPool", "multiprocessing.pool",
                        LimitedThreadPool27)
        elif sys.version_info.major == 3 and sys.version_info.minor >= 5:
            self._patch("Pool", "multiprocessing.pool", AffinityPool35)
            self._patch("ThreadPool", "multiprocessing.pool",
                        LimitedThreadPool35)
        if ProcessPoolExecutor:
            self._patch("ProcessPoolExecutor", "concurrent.futures",
                        AffinityProcessPoolExecutor)
        if ThreadPoolExecutor:
            self._patch("ThreadPoolExecutor", "concurrent.futures",
                        LimitedThreadPoolExecutor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self._items.keys():
            if self._items[name]:
                setattr(self._modules[name], name, self._items[name])

def _process_test(n):
    cpu_list = list(get_affinity())
    cpu_count = len(cpu_list)
    return cpu_count

def _test():
    target_factor = 1
    target_proc_num = 4
    success = True

    cpu_list = list(get_affinity())
    cpu_count = len(cpu_list)
    if cpu_count < target_factor:
        target_factor = cpu_count
    target_thread_num = target_factor
    if cpu_count >= target_proc_num:
        target_thread_num = target_thread_num*int(round(float(cpu_count)
                            /float(target_proc_num)))
    if not (sched_getaffinity or sched_setaffinity or PsutilProcess):
        target_thread_num = cpu_count

    with Monkey(oversubscription_factor = target_factor, max_top_workers = target_proc_num):
        p = getattr(__import__("multiprocessing.pool", globals(),  locals(), ["Pool"]), "Pool", None)()
        actual_thread_num = p.map(_process_test, [0 for i in range(target_proc_num)])
        for item in actual_thread_num:
            if item != target_thread_num:
                print("Expected thread number = {0}, actual = {1}".format(
                      target_thread_num, item))
                success = False
    if success:
        print("done")
    return 0 if success else 1

def _main():
    global oversubscription_factor
    global max_top_workers

    if not sys.platform.startswith('linux'):
        raise "Only linux is currently supported"

    import argparse
    parser = argparse.ArgumentParser(prog="python -m smp", description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--oversubscription-factor', default=oversubscription_factor, metavar='Number',
                        help="Limits maximal number of threads as available CPU * Number", type=int)
    parser.add_argument('-p', '--max-top-workers', default=max_top_workers, metavar='Number', type=int,
                        help="Limits outermost parallelism by controlling number of thread or "
                             "processes workers created by Python pools")
    parser.add_argument('-m', action='store_true', dest='module',
                        help="Executes following as a module")
    parser.add_argument('name', help="Script or module name")
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="Command line arguments")
    args = parser.parse_args()
    sys.argv = [args.name] + args.args
    if not os.environ.get("KMP_BLOCKTIME"):
        os.environ["KMP_BLOCKTIME"] = "0"
    if '_' + args.name in globals():
        return globals()['_' + args.name](*args.args)
    else:
        import runpy
        runf = runpy.run_module if args.module else runpy.run_path
        with Monkey(oversubscription_factor = args.oversubscription_factor,
                    max_top_workers = args.max_top_workers):
            runf(args.name, run_name='__main__')

