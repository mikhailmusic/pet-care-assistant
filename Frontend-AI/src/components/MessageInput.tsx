import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
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
  const [isRecording, setIsRecording] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingMimeRef = useRef<string | null>(null);

  const { addMessage, updateMessage, removeMessage, setMessages, isSending, setSending, currentChat } = useChatStore();

  const inferFileType = (mime: string): FileMetadata['file_type'] => {
    if (mime.startsWith('image/')) return 'image';
    if (mime.startsWith('video/')) return 'video';
    if (mime.startsWith('audio/')) return 'audio';
    return 'document';
  };

  const nextTempId = useMemo(() => Date.now() * -1, []);
  const preferredAudioMime = useMemo(() => {
    if (typeof MediaRecorder === 'undefined') return null;
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/wav'];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || null;
  }, []);

  const getAudioExtension = (mime: string | null) => {
    if (!mime) return 'webm';
    if (mime.includes('wav')) return 'wav';
    if (mime.includes('ogg')) return 'ogg';
    return 'webm';
  };

  const cleanupRecorder = () => {
    recordingChunksRef.current = [];
    recordingMimeRef.current = null;
    if (mediaRecorderRef.current?.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
    mediaRecorderRef.current = null;
  };

  useEffect(() => {
    return cleanupRecorder;
  }, []);

  const handleToggleRecording = async () => {
    try {
      if (isRecording) {
        mediaRecorderRef.current?.stop();
        setIsRecording(false);
        return;
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        alert('Браузер не поддерживает запись аудио');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = preferredAudioMime ? { mimeType: preferredAudioMime } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      const recorderMime = preferredAudioMime || mediaRecorder.mimeType || 'audio/webm';
      mediaRecorderRef.current = mediaRecorder;
      recordingChunksRef.current = [];
      recordingMimeRef.current = recorderMime;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const mimeType = recordingMimeRef.current || 'audio/webm';
        const blob = new Blob(recordingChunksRef.current, { type: mimeType });
        if (blob.size > 0) {
          const extension = getAudioExtension(mimeType);
          const file = new File([blob], `voice-${Date.now()}.${extension}`, { type: mimeType });
          setFiles((prev) => [...prev, file]);
        }
        cleanupRecorder();
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
      alert('Не удалось получить доступ к микрофону');
      cleanupRecorder();
      setIsRecording(false);
    }
  };

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
      // Перечитываем сообщения, чтобы временный id заменился на серверный
      const refreshed = await apiClient.getChatMessages(chatId);
      setMessages(refreshed);
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
            title="Добавить вложения"
          >
            +
          </button>
          <button
            type="button"
            className={`file-attach-btn ${isRecording ? 'recording' : ''}`}
            onClick={handleToggleRecording}
            disabled={isSending}
            title={isRecording ? 'Остановить запись' : 'Записать голосовое'}
          >
            {isRecording ? '■' : '🎤'}
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
            placeholder="Напишите сообщение или прикрепите файл"
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

        {currentChat && (
          <div className="inline-settings">
            <ComposerControls chat={currentChat} />
          </div>
        )}
      </div>
    </div>
  );
}
