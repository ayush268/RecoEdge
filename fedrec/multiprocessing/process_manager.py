from kafka import KafkaProducer
from fedrec.trainers.base_trainer import BaseTrainer
from collections import defaultdict
from typing import Any, Dict, List
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
from fedrec.utilities import registry
from fedrec.utilities.logger import BaseLogger
from fedrec.trainers.base_trainer import BaseTrainer
from fedrec.multiprocessing.job import Jobber
from kafka import KafkaConsumer
import json
from fedrec.communications.kafka_utils import publish_message
from mpi4py import MPI

class MPIProcessManager:

    def __init__(self, config) -> None:
        self.stream_config = config
        self.jobber = Jobber(trainer = registry.lookup("trainer"), logger = registry.lookup("logger"))
        self.consumer = KafkaConsumer(
            self.stream_config["consumer_topic"], bootstrap_servers=self.stream_config["bootstrap_servers"], auto_offset_reset="earliest", api_version=(0, 10))
        self.producer = KafkaProducer(bootstrap_servers = self.stream_config["bootstrap_servers"], api_version=(0,10))

    def consume(self):
        while True:
            with MPIPoolExecutor(max_workers=self.stream_config["num_processes"]) as executor:
                messages = self.consumer.poll(max_records=self.stream_config["consumer_max_records"])
                if len(messages.keys()) > 0:
                    for values in messages.values():
                        for message in values:
                            key = message.key.decode("utf-8")
                            message_dict = json.loads(message.value)
                            if key == "END":
                                executor.shutdown(wait=True)
                                return
                            else:
                                future = executor.submit(self.jobber.run, message_dict)
                                result_dict = future.result()
                                result_dict["worker_id"] = message_dict["worker_id"]
                                self.produce(result_dict)


    def produce(self, result):
        publish_message(self.producer, self.stream_config["producer_topic"], "job_result", json.dumps(result))
