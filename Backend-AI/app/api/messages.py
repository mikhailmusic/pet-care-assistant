from __future__ import annotations

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from app.dto import MessageResponseDTO, MessageCreateDTO, MessageUpdateDTO
from app.dependencies import CurrentUser
from app.dependencies import MessageServiceDep

router = APIRouter(tags=["Messages"])



class MetadataPatchRequest(BaseModel):
    patch: Dict[str, Any] = Field(default_factory=dict)


@router.get("/chats/{chat_id}/messages", response_model=List[MessageResponseDTO])
async def list_chat_messages(
    chat_id: int,
    current_user: CurrentUser,
    service: MessageServiceDep,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    order: str = Query(default="asc"),  # asc|desc
):
    return await service.list_chat_messages(
        chat_id=chat_id,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        order=order,
    )


@router.post("/chats/{chat_id}/messages", response_model=MessageResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_user_message(
    chat_id: int,
    payload: MessageCreateDTO,
    current_user: CurrentUser,
    service: MessageServiceDep,
):

    return await service.create_user_message(
        chat_id=chat_id,
        user_id=current_user.id,
        content=payload.content,
        file_ids=payload.files,
    )


@router.patch("/messages/{message_id}", response_model=MessageResponseDTO)
async def update_user_message(
    message_id: int,
    payload: MessageUpdateDTO,
    current_user: CurrentUser,
    service: MessageServiceDep,
):
    return await service.update_user_message(
        message_id=message_id,
        user_id=current_user.id,
        content=payload.content,
        file_ids=payload.files,
    )


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    current_user: CurrentUser,
    service: MessageServiceDep,
):
    ok = await service.delete_message(message_id=message_id, user_id=current_user.id)
    return {"success": ok}


@router.patch("/messages/{message_id}/metadata", response_model=MessageResponseDTO)
async def patch_message_metadata(
    message_id: int,
    payload: MetadataPatchRequest,
    current_user: CurrentUser,
    service: MessageServiceDep,
):
    return await service.add_metadata(
        message_id=message_id,
        user_id=current_user.id,
        patch=payload.patch,
    )
