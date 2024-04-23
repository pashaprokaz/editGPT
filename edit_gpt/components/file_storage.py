from abc import ABC, abstractmethod


class BaseFileReader(ABC):
    @abstractmethod
    def read_from_filename(self, filename):
        pass
