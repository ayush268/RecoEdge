import mpi4py
from fedrec.communications.process_com_manager import ProcessComManager
from fedrec.utilities import registry
from fedrec.multiprocessing.job import Jobber
from mpi4py import MPI
import asyncio
class MPIProcessManager:

    def __init__(self, config) -> None:
        self.pool = MPI.COMM_WORLD
        self.rank = self.pool.Get_rank()
        self.num_processes = self.pool.Get_size()
        self.jobber = Jobber(trainer = registry.lookup("trainer"), logger = registry.lookup("logger"))
        self.enqueued_jobs = []
        self.process_comm_manager = ProcessComManager(config_dict=config["comm_manager_config"])
        self.max_jobs_per_process = config["max_jobs_per_process"]
        self.loop = asyncio.get_event_loop()


    def run(self) -> None:
        self.loop.create_task(self.consume())
        self.loop.create_task(self.run_jobs())
        self.loop.run_forever()

    async def consume(self) -> None:
        while True:
            if len(self.enqueued_jobs) < self.max_jobs_per_process:
                job_request = self.process_comm_manager.receive_message()
                if job_request is not None:
                    if job_request["job_type"] == "END":
                        self.loop.stop()
                    self.enqueued_jobs.append(self,job_request)


    async def run_jobs(self) -> None:
        while True:
            if len(self.enqueued_jobs) > 0:
                job_request = self.enqueued_jobs.pop(0)
                job = self.loop.create_task(self.jobber.run(job_request))
                job.add_done_callback(self.publish())



    def publish(self, job_result) -> None:
        self.process_comm_manager.send_message(job_result.result())

    def balance_load(self):
        #TODO: Balance incoming task consumption between workers