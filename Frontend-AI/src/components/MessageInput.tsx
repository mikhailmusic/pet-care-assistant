import { useMemo, useState, type FormEvent, useRef, type ChangeEvent } from 'react';
import { apiClient, apiErrorMessage } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import type { ChatMessage, FileMetadata } from '../types';
import { ComposerControls } from './ComposerControls';
import './MessageInput.css';

interface MessageInputProps {
  chatId: number;
}

export function MessageInput({ chatId }: MessageInputProps) {
  const [content, setContent] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { addMessage, updateMessage, removeMessage, isSending, setSending, currentChat } = useChatStore();

  const inferFileType = (mime: string): FileMetadata['file_type'] => {
    if (mime.startsWith('image/')) return 'image';
    if (mime.startsWith('video/')) return 'video';
    if (mime.startsWith('audio/')) return 'audio';
    return 'document';
  };

  const nextTempId = useMemo(() => Date.now() * -1, []);

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault();

    if (!content.trim() && files.length === 0) return;

    setSending(true);

    const tempId = nextTempId + Math.floor(Math.random() * 1000);
    const optimisticFiles: FileMetadata[] = files.map((file) => ({
      file_id: `temp-${file.name}-${file.lastModified}`,
      filename: file.name,
      file_type: inferFileType(file.type),
      file_size: file.size,
      mime_type: file.type,
    }));

    const optimisticUser: ChatMessage = {
      id: tempId,
      chat_id: chatId,
      role: 'user',
      content: content.trim(),
      message_type: optimisticFiles.length ? inferFileType(optimisticFiles[0].mime_type) : 'text',
      files: optimisticFiles.length ? optimisticFiles : null,
      metadata_json: null,
      processing_time_ms: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      is_deleted: false,
    };

    addMessage(optimisticUser);

    try {
      const uploadedMetas: FileMetadata[] = [];
      const fileIds: string[] = [];
      for (const file of files) {
        const uploaded = await apiClient.uploadFile(file);
        fileIds.push(uploaded.file_id);
        uploadedMetas.push({
          file_id: uploaded.file_id,
          filename: file.name,
          file_type: inferFileType(file.type),
          file_size: file.size,
          mime_type: file.type,
          url: uploaded.url,
        });
      }

      const assistantMessage = await apiClient.sendMessage(chatId, {
        content: content.trim(),
        files: fileIds.length > 0 ? fileIds : undefined,
      });

      updateMessage(tempId, { files: uploadedMetas.length ? uploadedMetas : null });

      addMessage(assistantMessage);
      setContent('');
      setFiles([]);
    } catch (error) {
      console.error('Failed to send message:', error);
      removeMessage(tempId);
      alert(apiErrorMessage(error));
    } finally {
      setSending(false);
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  return (
    <div className="message-input-container">
      <div className="message-input-inner">
        <div className="input-toolbar">
          <button
            type="button"
            className="settings-toggle"
            onClick={() => setShowSettings((prev) => !prev)}
            aria-expanded={showSettings}
          >
            ⚙️ Параметры
          </button>
          <span className="toolbar-hint">Shift+Enter — перенос строки</span>
        </div>

        {showSettings && currentChat && (
          <div className="inline-settings">
            <ComposerControls chat={currentChat} />
          </div>
        )}

        {files.length > 0 && (
          <div className="selected-files">
            {files.map((file, idx) => (
              <div key={idx} className="file-chip">
                <span>{file.name}</span>
                <button type="button" onClick={() => handleRemoveFile(idx)}>
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <form className="message-input-form" onSubmit={handleSubmit}>
          <button
            type="button"
            className="file-attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Прикрепить файлы"
          >
            📎
          </button>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            accept="image/*,video/*,audio/*,.pdf,.docx,.txt,.csv,.xlsx"
          />

          <textarea
            className="message-textarea"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Спросите про уход, вакцинацию, рацион или прикрепите файлы…"
            rows={1}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void handleSubmit();
              }
            }}
          />

          <button type="submit" className="send-btn" disabled={isSending || (!content.trim() && files.length === 0)}>
            {isSending ? 'Отправляем…' : 'Отправить'}
          </button>
        </form>
      </div>
    </div>
  );
}
