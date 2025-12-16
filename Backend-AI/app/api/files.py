# app/api/routes/files.py

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Query, status

from app.dto import FileUploadResponseDTO, MultipleFileUploadResponseDTO, FileMetadataDTO
from app.dependencies import CurrentUser
from app.dependencies import FileServiceDep

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload", response_model=FileUploadResponseDTO, status_code=status.HTTP_201_CREATED)
async def upload_file(
    current_user: CurrentUser,
    service: FileServiceDep,
    file: UploadFile = File(...),
    folder: str = Query(default="uploads"),
):
    return await service.upload_file(user_id=current_user.id, file=file, folder=folder)


@router.post("/upload-multiple", response_model=MultipleFileUploadResponseDTO, status_code=status.HTTP_201_CREATED)
async def upload_files(
    current_user: CurrentUser,
    service: FileServiceDep,
    files: List[UploadFile] = File(...),
    folder: str = Query(default="uploads"),
):
    return await service.upload_files(user_id=current_user.id, files=files, folder=folder)


@router.get("/{file_id}", response_model=FileMetadataDTO)
async def get_file_metadata(
    file_id: str,
    current_user: CurrentUser,
    service: FileServiceDep,
):
    return await service.get_file_metadata(user_id=current_user.id, file_id=file_id)


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: CurrentUser,
    service: FileServiceDep,
):
    ok = await service.delete_file(user_id=current_user.id, file_id=file_id)
    return {"success": ok}
