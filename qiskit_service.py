from qiskit_ibm_runtime import QiskitRuntimeService
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('IBM_QUANTUM_TOKEN')
instance = os.getenv('IBM_QUANTUM_INSTANCE')

if token is None:
    raise ValueError("Please set IBM_QUANTUM_TOKEN in your .env file")

QiskitRuntimeService.save_account(
    token=token,
    instance=instance,
    overwrite=True
)

service = QiskitRuntimeService()