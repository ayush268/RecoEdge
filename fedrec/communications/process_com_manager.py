
from fedrec.communications.messages import JobMessage
from fedrec.communications.comm_manager import (CommunicationManager,
                                                tag_reciever)


class ProcessComManager(CommunicationManager):
    def __init__(self, config_dict):
        super().__init__(config_dict=config_dict)

    def run(self):
        super().run()

    @tag_reciever(JobMessage.JOB_REQUEST)
    def send_message(self, msg_params):
        #TODO : Send job completion message to worker comm manager.
        raise NotImplementedError

    @tag_reciever(JobMessage.JOB_COMPLETION)
    async def receive_message(self,):
        #TODO : Distributed consumption of job requests
        raise NotImplementedError
