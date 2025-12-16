from __future__ import annotations

from typing import List
from fastapi import APIRouter, Query, status

from app.dto import ChatCreateDTO, ChatUpdateDTO, ChatResponseDTO, ChatListItemDTO
from app.dto import ChatSettingsDTO
from app.dependencies import CurrentUser
from app.dependencies import ChatServiceDep

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.post("", response_model=ChatResponseDTO, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: ChatCreateDTO,
    current_user: CurrentUser,
    service: ChatServiceDep,
):
    return await service.create_chat(user_id=current_user.id, dto=payload)


@router.get("", response_model=List[ChatListItemDTO])
async def list_user_chats(
    current_user: CurrentUser,
    service: ChatServiceDep,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
):
    return await service.list_user_chats(user_id=current_user.id, skip=skip, limit=limit)


@router.get("/{chat_id}", response_model=ChatResponseDTO)
async def get_chat(
    chat_id: int,
    current_user: CurrentUser,
    service: ChatServiceDep,
    with_messages: bool = Query(default=False),
):
    return await service.get_chat(chat_id=chat_id, user_id=current_user.id, with_messages=with_messages)


@router.patch("/{chat_id}", response_model=ChatResponseDTO)
async def update_chat(
    chat_id: int,
    payload: ChatUpdateDTO,
    current_user: CurrentUser,
    service: ChatServiceDep,
):
    return await service.update_chat(chat_id=chat_id, user_id=current_user.id, dto=payload)


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: int,
    current_user: CurrentUser,
    service: ChatServiceDep,
):
    ok = await service.delete_chat(chat_id=chat_id, user_id=current_user.id)
    return {"success": ok}


@router.get("/{chat_id}/settings", response_model=ChatSettingsDTO)
async def get_chat_settings(
    chat_id: int,
    current_user: CurrentUser,
    service: ChatServiceDep,
):
    return await service.get_chat_settings(chat_id=chat_id, user_id=current_user.id)
