from abc import ABC, abstractmethod

class RunnerAPI(ABC):
  def __init__(
      self,
      response_timeout: int = 90,
  ):
    self.response_timeout = response_timeout


  @abstractmethod
  async def run(self, host: str = "0.0.0.0", port: int = 52415):
    pass
