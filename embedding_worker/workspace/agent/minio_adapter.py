import os
from minio.error import S3Error
from minio import Minio
from io import BytesIO,StringIO
import pathlib
from pathlib import Path
import aiofiles
from logger import logger

class MinioAdapter:
    def __init__(self, minio_endpoint: str, access_key: str, secret_key: str):
        self._client = Minio(
            minio_endpoint,
            access_key = access_key,
            secret_key = secret_key,
            secure = False
        )
        
    def read_file_from_minio(self, bucket: str, object_name: str):
        try:
            data = self._client.get_object(bucket, object_name)
            return data.read()
        except S3Error as e:
            logger.error(f"Error in reading file from minio | {e}")
            return None
    
    # async def store_locally_async(self, file_bytes: bytes, dest_path: str) -> None:
    #     pathlib.Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    #     async with aiofiles.open(dest_path, "wb") as f:
    #         await f.write(file_bytes)
        
    def store_locally_sync(self, file_bytes: bytes, dest_path: str) -> None:
        try:
            p = Path(dest_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(file_bytes)
            return True
        except Exception as e:
            logger.error(f"Error in saving file {e}")
            return False 