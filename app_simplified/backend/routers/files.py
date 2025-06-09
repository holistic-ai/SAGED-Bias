import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app_simplified.backend.services.file_service import FileService

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

# Create a single instance of FileService
file_service = FileService()

@router.post("/upload/{domain}")
async def upload_file(domain: str, file: UploadFile = File(...)):
    """Upload a file for a specific domain"""
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
        # Save the file
        file_path = await file_service.save_uploaded_file(file, domain)
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "data": {
                "filename": file.filename,
                "path": file_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{domain}")
async def get_domain_files(domain: str):
    """Get list of files for a domain"""
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
    """Get full paths of all files for a domain"""
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
    """Get content of a specific file"""
    try:
        file_path = file_service.get_upload_path(domain)
        full_path = os.path.join(file_path, filename)
        content = file_service.get_file_content(full_path)
        return {
            "status": "success",
            "data": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{domain}")
async def cleanup_domain_files(domain: str):
    """Clean up all files for a domain"""
    try:
        success = file_service.cleanup_domain_files(domain)
        if success:
            return {
                "status": "success",
                "message": "Domain files cleaned up successfully"
            }
        raise HTTPException(status_code=404, detail="Domain directory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 