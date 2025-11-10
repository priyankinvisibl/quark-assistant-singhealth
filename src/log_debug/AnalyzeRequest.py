from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    error_log: str
    pipeline_type: str = "Nextflow"  # Optional with default
