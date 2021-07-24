
from enum import Enum

class ProcMessage(Enum):
    SYNC_MODEL = 1

class JobMessage():
    JOB_REQUEST = {}
    JOB_COMPLETION = {}
