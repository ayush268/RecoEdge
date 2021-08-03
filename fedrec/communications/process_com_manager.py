
from fedrec.communications.messages import JobMessage
from fedrec.communications.comm_manager import (CommunicationManager,
                                                tag_reciever)


class ProcessComManager(CommunicationManager):
    def __init__(self, config_dict):
        super().__init__(config_dict=config_dict)

    def run(self):
        super().run()


    def handle_message(self):
        #TODO : Distributed consumption of job requests
        raise NotImplementedError