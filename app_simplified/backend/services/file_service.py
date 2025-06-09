import os
import shutil
from datetime import datetime
import logging
from typing import List, Optional
from fastapi import UploadFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FileService')

class FileService:
    def __init__(self):
        # Create uploads directory if it doesn't exist
        self.uploads_dir = os.path.abspath("data/uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)
        logger.info(f"FileService initialized with uploads directory: {self.uploads_dir}")

    def get_upload_path(self, domain: str) -> str:
        """Get the upload path for a specific domain"""
        domain_dir = os.path.join(self.uploads_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        return domain_dir

    async def save_uploaded_file(self, file: UploadFile, domain: str) -> str:
        """
        Save an uploaded file to the domain's directory
        
        Args:
            file: The uploaded file
            domain: The domain name
            
        Returns:
            str: The path where the file was saved
        """
        try:
            # Create domain-specific directory
            domain_dir = self.get_upload_path(domain)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(domain_dir, filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved successfully: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise Exception(f"Failed to save file: {str(e)}")

    def get_domain_files(self, domain: str) -> List[str]:
        """Get list of files for a specific domain"""
        try:
            domain_dir = self.get_upload_path(domain)
            files = [f for f in os.listdir(domain_dir) if os.path.isfile(os.path.join(domain_dir, f))]
            return files
        except Exception as e:
            logger.error(f"Error getting domain files: {str(e)}")
            return []

    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted successfully: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False

    def cleanup_domain_files(self, domain: str) -> bool:
        """Clean up all files for a domain"""
        try:
            domain_dir = self.get_upload_path(domain)
            if os.path.exists(domain_dir):
                shutil.rmtree(domain_dir)
                logger.info(f"Domain files cleaned up successfully: {domain_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up domain files: {str(e)}")
            return False

    def get_file_content(self, file_path: str) -> str:
        """Read the content of a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise Exception(f"Failed to read file: {str(e)}")

    def get_domain_file_paths(self, domain: str) -> List[str]:
        """Get full paths of all files for a domain"""
        try:
            domain_dir = self.get_upload_path(domain)
            return [os.path.join(domain_dir, f) for f in os.listdir(domain_dir) 
                   if os.path.isfile(os.path.join(domain_dir, f))]
        except Exception as e:
            logger.error(f"Error getting domain file paths: {str(e)}")
            return [] 