import { useMemo, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { ru } from 'date-fns/locale';
import ReactMarkdown from 'react-markdown';
import { apiClient, apiErrorMessage } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import type { ChatMessage, FileMetadata } from '../types';
import './MessageItem.css';

interface MessageItemProps {
  message: ChatMessage;
}

export function MessageItem({ message }: MessageItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);
  const [editedFiles, setEditedFiles] = useState<FileMetadata[]>(message.files || []);
  const [isUpdating, setIsUpdating] = useState(false);

  const { updateMessage, addMessage, messages, removeMessage } = useChatStore();

  const timeAgo = useMemo(
    () =>
      formatDistanceToNow(new Date(message.created_at), {
        addSuffix: true,
        locale: ru,
      }),
    [message.created_at]
  );

  const handleEdit = () => {
    setIsEditing(true);
    setEditedContent(message.content);
    setEditedFiles(message.files || []);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditedContent(message.content);
    setEditedFiles(message.files || []);
  };

  const handleSave = async () => {
    const trimmed = editedContent.trim();
    if (!trimmed && editedFiles.length === 0) {
      alert('Нельзя сохранить пустое сообщение.');
      return;
    }
    const originalFiles = message.files || [];
    const isSameFiles =
      originalFiles.length === editedFiles.length &&
      originalFiles.every((file) => editedFiles.some((edited) => edited.file_id === file.file_id));

    if (trimmed === message.content && isSameFiles) {
      setIsEditing(false);
      return;
    }

    setIsUpdating(true);

    try {
      const removedFiles = (message.files || []).filter(
        (file) => !editedFiles.some((edited) => edited.file_id === file.file_id)
      );

      const newAssistantMessage = await apiClient.updateMessage(message.id, {
        content: trimmed,
        files: editedFiles.map((file) => file.file_id),
      });

      for (const file of removedFiles) {
        try {
          await apiClient.deleteFile(file.file_id);
        } catch (err) {
          console.warn('Failed to delete file', file.file_id, err);
        }
      }

      updateMessage(message.id, { content: trimmed, files: editedFiles, isEditing: false });

      const messageIndex = messages.findIndex((m) => m.id === message.id);
      if (messageIndex !== -1) {
        const subsequentMessages = messages.slice(messageIndex + 1);
        subsequentMessages.forEach((m) => {
          updateMessage(m.id, { is_deleted: true });
        });
      }

      addMessage(newAssistantMessage);
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update message:', error);
      alert(apiErrorMessage(error));
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Удалить сообщение и его вложения?')) return;

    try {
      await apiClient.deleteMessage(message.id);
      if (message.files) {
        for (const file of message.files) {
          try {
            await apiClient.deleteFile(file.file_id);
          } catch (err) {
            console.warn('Failed to delete file', file.file_id, err);
          }
        }
      }
      removeMessage(message.id);
    } catch (error) {
      console.error('Failed to delete message:', error);
      alert(apiErrorMessage(error));
    }
  };

  if (message.is_deleted) {
    return null;
  }

  return (
    <div className={`message-item ${message.role}`}>
      <div className="message-avatar">{message.role === 'user' ? '🙂' : '🤖'}</div>

      <div className="message-content-wrapper">
        <div className="message-header">
          <span className="message-role">{message.role === 'user' ? 'Вы' : 'Ассистент'}</span>
          <span className="message-time">{timeAgo}</span>
        </div>

        <div className="message-content">
          {isEditing ? (
            <div className="message-edit">
              <textarea
                value={editedContent}
                onChange={(e) => setEditedContent(e.target.value)}
                className="message-edit-textarea"
                rows={4}
                autoFocus
              />

              {editedFiles.length > 0 && (
                <div className="message-files editing">
                  {editedFiles.map((file) => (
                    <div key={file.file_id} className="file-attachment">
                      <span>{file.filename}</span>
                      <button type="button" onClick={() => setEditedFiles((prev) => prev.filter((f) => f.file_id !== file.file_id))}>
                        Удалить
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="message-edit-actions">
                <button onClick={handleSave} disabled={isUpdating} className="btn-save">
                  {isUpdating ? 'Сохраняем…' : 'Сохранить и пересчитать ответ'}
                </button>
                <button onClick={handleCancel} disabled={isUpdating} className="btn-cancel">
                  Отмена
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="message-text">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>

              {message.files && message.files.length > 0 && (
                <div className="message-files">
                  {message.files.map((file) => (
                    <div key={file.file_id} className="file-attachment">
                      {file.file_type === 'image' && file.url ? (
                        <img src={file.url} alt={file.filename} className="file-image" />
                      ) : (
                        <a href={file.url} target="_blank" rel="noopener noreferrer" className="file-link">
                          📎 {file.filename}
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {message.processing_time_ms && (
                <div className="message-meta">
                  <small>Ответ подготовлен за {message.processing_time_ms} мс</small>
                </div>
              )}
            </>
          )}
        </div>

        {!isEditing && message.role === 'user' && (
          <div className="message-actions">
            <button onClick={handleEdit} className="btn-icon" title="Редактировать">
              ✎
            </button>
            <button onClick={handleDelete} className="btn-icon" title="Удалить">
              🗑
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
