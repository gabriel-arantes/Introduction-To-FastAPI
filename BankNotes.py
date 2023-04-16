from pydantic import BaseModel

# 1. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float  