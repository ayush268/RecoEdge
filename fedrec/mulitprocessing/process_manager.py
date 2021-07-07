from RecoEdge.fedrec.trainers.base_trainer import BaseTrainer
from collections import defaultdict
from typing import Any, Dict, List
from mpi4py import MPI
from fedrec.utilities import registry
from fedrec.utilities.logger import BaseLogger
from fedrec.trainers.base_trainer import BaseTrainer
from fedrec.mulitprocessing.job import Jobber
import asyncio
import itertools

class MPIProcessManager:

    def __init__(self, config, com_manager) -> None:
        self.pool = MPI.COMM_WORLD
        self.rank = self.pool.Get_rank()
        self.num_processes = self.pool.Get_size()

        # This can be substituted by MPIPoolProcess for only process execution
        # https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor

        if self.rank == 0:
            self.job_pool = []
            # jobs are worker -> job_type -> neighbours_out
            # jobs are worker -> storage
            # Comm Manager does this shouldn't do this in process manager
            # TODO : Multiple aggregators for same job
            self.enqueued_jobs = defaultdict(defaultdict(set))
            self.stream_pos = 0

            self.process_batch_size = config["process_batch_size"]
            self.read_stream_timeout = config["read_stream_timeout"]
            self.com_manager = com_manager

        # Define a jobber for each process
        self.jobber = Jobber(trainer=registry.lookup(
            BaseTrainer, "trainer"), logger=registry.lookup(BaseLogger, "logger"))

    async def run(self):
        while True:
            chunks = []
            if self.rank == 0:
                asyncio.wait_for(self.read_job_stream(),
                                 self.read_stream_timeout)
                chunks = self.create_process_pool()
                assert len(chunks) == len(self.num_processes)

            if len(chunks) == 0:
                continue
            chunk = self.pool.scatter(chunks, root=0)
            results = self.process_pool(chunk)
            results = self.pool.gather(results, root=0)

            if self.rank == 0:
                results = list(itertools.chain(*results))
                self.send_notifications(results)

    def process_pool(self, chunk: List[Any]):
        results = []
        for job in chunk:
            result = self.jobber.run(**job["train_params"])
            result["sender_id"] = job["sender_id"]
            result["worker_id"] = job["worker_id"]
        return results

    # This should be done in comm manager
    # But this logic needs to be checked. No retraining per round
    def remap_worker_tasks(self, role, reciever_id, sender_worker_ids):
        assert self.rank == 0, "Worker orchestration called from child process"
        for id in sender_worker_ids:
            self.enqueued_jobs[id][role].add(reciever_id)

    def create_process_pool(self):
        """
        Load worker states and split to chunks
        """
        assert self.rank == 0, "Main process only connected to comm manager"
        process_pool = []
        for job in self.job_pool:
            receiver_id, worker_id = job["receiver_id"], job["worker_id"]
            worker = self.com_manager.get_worker_by_id(worker_id)
            worker_state = worker.serialise()
            model, optimizer, model_preproc = worker_state.state_dict[
                "model"], worker_state.state_dict["optimizer"], worker_state.model_preproc
            train_params = {"model_state": model, "optimizer_state": optimizer,
                            "model_preproc": model_preproc}
            process_pool.append(
                {"worker_id": worker_id, "receiver_id": receiver_id, "train_params": train_params})

        (num_chunks, modulus) = divmod(len(process_pool), self.num_processes)
        if num_chunks == 0:
            chunks = [[chunk] for chunk in process_pool]
            ## Add empty lists to keep length num_processes
            for _ in range(self.num_processes - num_chunks):
                chunks.append([])
        else:
            chunks = (process_pool[i*num_chunks+min(i, modulus):(i+1) *
                      num_chunks+min(i+1, modulus)] for i in range(self.num_processes))
        self.job_pool = []
        return chunks


    def read_job_stream(self):
        # TODO : Read from stream tasks with order
        assert self.rank == 0, "Main process always reads streams"
        while True:
            job_stream, curr_pos = self.com_manager.read_job_stream()
            self.stream_pos = curr_pos
            self.job_pool += job_stream
            if len(self.job_pool) >= self.num_processes * self.process_batch_size:
                break
        return

    def send_notifications(self, results):
        assert self.rank == 0, "Main process handles all comms"
        for res in results:
            worker_id, receiver_id = res["worker_id"],res["receiver_id"]
            if res["status"]:
                self.com_manager.send_trained_response(worker_state=res["worker_state"], worker_id=worker_id, receiver_id = receiver_id)
                self.logger.info("Worker {} trained successfully for receiver {}".format(worker_id, receiver_id))
            else:
                self.logger.info("Worker {} DID NOT TRAIN for receiver {}, Error : {}".format(worker_id, receiver_id,res["error"]))

