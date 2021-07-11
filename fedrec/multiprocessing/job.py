from fedrec.trainers.base_trainer import BaseTrainer


class Jobber:
    """
    Jobber class only handles job requests based on job type
    """
    def __init__(self, trainer, logger) -> None:
        self.logger = logger
        self.trainer: BaseTrainer = trainer

    def run(self, message_dict):
        print("here")
        job_type = message_dict["job_type"]
        if job_type == "train":
            try:
                worker_state = message_dict["worker_state"]
                self.trainer.load_state(worker_state)
                trained_state = self.trainer.train(self.logger)
                ## Trainer should return state of trained model
                ## Or do you want to run a worker-specific model
                return {"worker_state" : trained_state, "status":"done"}
            except Exception as e:
                return {"status": "fail", "error": str(e)}
        else:
            return ValueError("Current implementation only for train")