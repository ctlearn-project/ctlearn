from enum import Enum

class FrameworkType(Enum):
    KERAS = 1
    PYTORCH = 2


class Task(Enum):
    type = 0
    energy = 1
    direction = 2
    cameradirection = 3
    skydirection = 4
    all = 5

class EventType(Enum):
    gamma=0
    proton=1
    electron=2
    
class Mode(Enum):
    
    train = 0
    results = 1
    validate = 2
    observation = 3
    tunning = 4
    