from pydantic import BaseModel

class SnapshotRequest(BaseModel):
    user: str
