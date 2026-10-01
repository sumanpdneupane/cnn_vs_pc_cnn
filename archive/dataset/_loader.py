from abc import ABC, abstractmethod

class ImageNetLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self) -> tuple:
        pass