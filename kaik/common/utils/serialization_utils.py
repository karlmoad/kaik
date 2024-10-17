from joblib import load, dump
from io import BytesIO
import base64


def serialize(obj:object) -> str:
    buffer = BytesIO()
    dump(obj, buffer)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def deserialize(serialized:str) -> object:
    ser_bytes = serialized.encode('ascii')
    return load(BytesIO(base64.b64decode(ser_bytes)))