import logging
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
from sqlalchemy import create_engine, text
from app_simplified.backend.schemas.build_config import DatabaseConfig
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FileService')

class FileService:
    def __init__(self, database_config: Optional[DatabaseConfig] = None):
        # Initialize database configuration
        self.database_config = database_config or DatabaseConfig()
        if not self.database_config.use_database:
            raise ValueError("Database must be enabled for FileService")
        
        if self.database_config.database_type != 'sql':
            raise ValueError("Only SQL database type is supported")
            
        self.engine = create_engine(self.database_config.database_connection)
        self._ensure_source_text_table()
        logger.info(f"FileService initialized with database: {self.database_config.database_connection}")

    def _ensure_source_text_table(self):
        """Ensure the source text table exists in the database"""
        try:
            table_name = self.database_config.source_text_table
            with self.engine.connect() as conn:
                # Create table if it doesn't exist
                create_table_query = text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        domain TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(create_table_query)
                conn.commit()
            logger.info(f"Ensured source text table exists: {table_name}")
        except Exception as e:
            logger.error(f"Error ensuring source text table: {str(e)}")
            raise

    async def save_uploaded_file(self, file: UploadFile, domain: str) -> str:
        """
        Save an uploaded file's content to the database
        
        Args:
            file: The uploaded file
            domain: The domain name
            
        Returns:
            str: The file path used as the identifier in the database
        """
        try:
            # Generate unique identifier with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{domain}/{timestamp}_{file.filename}"
            
            # Read file content
            content = await file.read()
            content_str = content.decode('utf-8')
            
            # Store in database
            with self.engine.connect() as conn:
                query = text(f"""
                    INSERT OR REPLACE INTO {self.database_config.source_text_table} 
                    (file_path, content, domain)
                    VALUES (:file_path, :content, :domain)
                """)
                conn.execute(query, {
                    "file_path": file_path,
                    "content": content_str,
                    "domain": domain
                })
                conn.commit()
            
            # Enhanced logging with content preview
            content_preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
            logger.info(f"File saved to database - Table: {self.database_config.source_text_table}, "
                       f"Path: {file_path}, Content Preview: {content_preview}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file content: {str(e)}")
            raise Exception(f"Failed to save file content: {str(e)}")

    def get_file_content(self, file_path: str) -> str:
        """Retrieve text content from the database using file_path as identifier"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT content FROM {self.database_config.source_text_table} WHERE file_path = :file_path")
                result = conn.execute(query, {"file_path": file_path}).first()
                if result:
                    return result[0]
                raise FileNotFoundError(f"No content found for file path: {file_path}")
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            raise Exception(f"Failed to retrieve content: {str(e)}")

    def get_domain_files(self, domain: str) -> List[str]:
        """Get list of file paths for a specific domain from the database"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT file_path FROM {self.database_config.source_text_table} WHERE domain = :domain")
                results = conn.execute(query, {"domain": domain}).fetchall()
                return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting domain files: {str(e)}")
            return []

    def delete_file(self, file_path: str) -> bool:
        """Delete a file's content from the database"""
        try:
            with self.engine.connect() as conn:
                query = text(f"DELETE FROM {self.database_config.source_text_table} WHERE file_path = :file_path")
                result = conn.execute(query, {"file_path": file_path})
                conn.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"File content deleted from database: {file_path}")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting file content: {str(e)}")
            return False

    def cleanup_domain_files(self, domain: str) -> bool:
        """Clean up all files for a domain from the database"""
        try:
            with self.engine.connect() as conn:
                query = text(f"DELETE FROM {self.database_config.source_text_table} WHERE domain = :domain")
                result = conn.execute(query, {"domain": domain})
                conn.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Domain files cleaned up from database: {domain}")
                return deleted
        except Exception as e:
            logger.error(f"Error cleaning up domain files: {str(e)}")
            return False

    def get_domain_file_paths(self, domain: str) -> List[str]:
        """Get all file paths for a domain from the database"""
        return self.get_domain_files(domain) 