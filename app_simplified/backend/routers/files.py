from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List
from app_simplified.backend.services.file_service import FileService
from app_simplified.backend.schemas.build_config import DatabaseConfig, FileServiceConfig
from app_simplified.backend.services.database_service import DatabaseService
from pydantic import BaseModel

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

class TableNameUpdate(BaseModel):
    table_name: str

# Create instances of services
database_service = DatabaseService()
file_service_config = FileServiceConfig(
    database_config=DatabaseConfig(**database_service.get_database_config())
)
file_service = FileService(file_service_config.database_config)

@router.put("/config/table")
async def update_source_table(table_update: TableNameUpdate):
    """Update the source text table name in the database configuration"""
    try:
        # Update the file service config
        file_service_config.update_table(table_update.table_name)
        
        # Reinitialize the file service with new config
        global file_service
        file_service = FileService(file_service_config.database_config)
        
        return {
            "status": "success",
            "message": f"Source text table updated to: {table_update.table_name}",
            "data": {
                "table_name": file_service_config.current_table
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/table")
async def get_source_table():
    """Get the current source text table name from the database configuration"""
    try:
        return {
            "status": "success",
            "data": {
                "table_name": file_service_config.current_table
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/{domain}")
async def upload_file(domain: str, file: UploadFile = File(...)):
    """Upload a file's content to the database for a specific domain"""
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
        # Save the file content to database
        file_path = await file_service.save_uploaded_file(file, domain)
        
        return {
            "status": "success",
            "message": "File content saved successfully",
            "data": {
                "filename": file.filename,
                "path": file_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{domain}")
async def get_domain_files(domain: str):
    """Get list of file paths for a domain from the database"""
    try:
        files = file_service.get_domain_files(domain)
        return {
            "status": "success",
            "data": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{domain}/paths")
async def get_domain_file_paths(domain: str):
    """Get all file paths for a domain from the database"""
    try:
        paths = file_service.get_domain_file_paths(domain)
        return {
            "status": "success",
            "data": paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/{domain}/{filename}")
async def get_file_content(domain: str, filename: str):
    """Get content of a specific file from the database"""
    try:
        # Construct the file path as stored in the database
        file_path = f"{domain}/{filename}"
        content = file_service.get_file_content(file_path)
        return {
            "status": "success",
            "data": content
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{domain}")
async def cleanup_domain_files(domain: str):
    """Clean up all files for a domain from the database"""
    try:
        success = file_service.cleanup_domain_files(domain)
        if success:
            return {
                "status": "success",
                "message": "Domain files cleaned up successfully"
            }
        raise HTTPException(status_code=404, detail="No files found for domain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{domain}/{filename}")
async def delete_file(domain: str, filename: str):
    """Delete a specific file from the database"""
    try:
        file_path = f"{domain}/{filename}"
        success = file_service.delete_file(file_path)
        if success:
            return {
                "status": "success",
                "message": "File deleted successfully"
            }
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/sourcetable")
async def get_database_source_table():
    """Get the current source text table name from DatabaseService"""
    try:
        return {
            "status": "success",
            "data": {
                "table_name": DatabaseService.source_text_table
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/database/sourcetable")
async def update_database_source_table(table_update: TableNameUpdate):
    """Update the source text table name in DatabaseService"""
    try:
        # Update the class variable
        DatabaseService.source_text_table = table_update.table_name
        
        return {
            "status": "success",
            "message": f"Database source text table updated to: {table_update.table_name}",
            "data": {
                "table_name": DatabaseService.source_text_table
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 