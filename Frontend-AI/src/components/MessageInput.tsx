import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
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
  const [files, setFiles] = useState<Array<{ file: File; previewUrl: string }>>([]);
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

  const audioBufferToWav = (audioBuffer: AudioBuffer) => {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataLength = audioBuffer.length * blockAlign;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    let offset = 0;
    const writeString = (str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
      offset += str.length;
    };

    writeString('RIFF');
    view.setUint32(offset, 36 + dataLength, true);
    offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, format, true);
    offset += 2;
    view.setUint16(offset, numChannels, true);
    offset += 2;
    view.setUint32(offset, sampleRate, true);
    offset += 4;
    view.setUint32(offset, sampleRate * blockAlign, true);
    offset += 4;
    view.setUint16(offset, blockAlign, true);
    offset += 2;
    view.setUint16(offset, bitDepth, true);
    offset += 2;
    writeString('data');
    view.setUint32(offset, dataLength, true);
    offset += 4;

    const channelData: Float32Array[] = [];
    for (let channel = 0; channel < numChannels; channel++) {
      channelData.push(audioBuffer.getChannelData(channel));
    }

    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
      }
    }

    return buffer;
  };

  const blobToWavFile = async (blob: Blob) => {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new AudioContext();
    try {
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const wavBuffer = audioBufferToWav(decoded);
      const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      return new File([wavBlob], `voice-${Date.now()}.wav`, { type: 'audio/wav' });
    } finally {
      await audioContext.close();
    }
  };

  useEffect(() => {
    return () => {
      cleanupRecorder();
    };
  }, []);

  const addFilesWithPreview = useCallback((newFiles: File[]) => {
    setFiles((prev) => [
      ...prev,
      ...newFiles.map((file) => ({
        file,
        previewUrl: URL.createObjectURL(file),
      })),
    ]);
  }, []);

  useEffect(() => {
    const listener = (event: Event) => {
      const detail = (event as CustomEvent<File[]>).detail;
      if (detail?.length) {
        addFilesWithPreview(detail);
      }
    };
    window.addEventListener('chat-files-dropped', listener as EventListener);
    return () => {
      window.removeEventListener('chat-files-dropped', listener as EventListener);
    };
  }, [addFilesWithPreview]);

  const removeFileAt = (index: number) => {
    setFiles((prev) => {
      const target = prev[index];
      if (target) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleToggleRecording = async () => {
    try {
      if (isRecording) {
        mediaRecorderRef.current?.stop();
        setIsRecording(false);
        return;
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        alert('Ваш браузер не поддерживает запись аудио.');
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
        void (async () => {
          try {
            if (blob.size > 0) {
              const wavFile = await blobToWavFile(blob);
              addFilesWithPreview([wavFile]);
            }
          } catch (conversionError) {
            console.warn('Falling back to recorded audio blob:', conversionError);
            if (blob.size > 0) {
              const extension = getAudioExtension(mimeType);
              const file = new File([blob], `voice-${Date.now()}.${extension}`, { type: mimeType });
              addFilesWithPreview([file]);
            }
          } finally {
            cleanupRecorder();
          }
        })();
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
      alert('Не удалось начать запись. Проверьте разрешения на микрофон.');
      cleanupRecorder();
      setIsRecording(false);
    }
  };

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault();

    if (!content.trim() && files.length === 0) return;

    setSending(true);

    const tempId = nextTempId + Math.floor(Math.random() * 1000);
    const optimisticFiles: FileMetadata[] = files.map(({ file, previewUrl }) => ({
      file_id: `temp-${file.name}-${file.lastModified}`,
      filename: file.name,
      file_type: inferFileType(file.type),
      file_size: file.size,
      mime_type: file.type,
      url: previewUrl,
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
      for (const { file } of files) {
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
      // чтобы синхронизировать сообщения, если id ассистента отличаются от локальных
      const refreshed = await apiClient.getChatMessages(chatId);
      setMessages(refreshed);
      setContent('');
      files.forEach(({ previewUrl }) => URL.revokeObjectURL(previewUrl));
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
      addFilesWithPreview(Array.from(e.target.files));
    }
  };

  return (
    <div className="message-input-container">
      <div className="message-input-inner">
        {files.length > 0 && (
          <div className="selected-files">
            {files.map(({ file, previewUrl }, idx) => {
              const isAudio = file.type.startsWith('audio/');
              return (
                <div key={`${file.name}-${file.lastModified}-${idx}`} className="file-chip">
                  <div className="file-chip__info">
                    <span>{file.name}</span>
                    {isAudio && (
                      <audio controls src={previewUrl} className="file-chip__audio">
                        Your browser does not support the audio element.
                      </audio>
                    )}
                  </div>
                  <button type="button" onClick={() => removeFileAt(idx)}>
                    ×
                  </button>
                </div>
              );
            })}
          </div>
        )}

        <form className="message-input-form" onSubmit={handleSubmit}>
          <button
            type="button"
            className="file-attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Прикрепить файлы"
          >
            +
          </button>
          <button
            type="button"
            className={`file-attach-btn ${isRecording ? 'recording' : ''}`}
            onClick={handleToggleRecording}
            disabled={isSending}
            title={isRecording ? 'Остановить запись' : 'Начать запись голоса'}
          >
            {isRecording ? '■' : '⏺'}
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
            placeholder="Введите сообщение или прикрепите файлы"
            rows={1}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void handleSubmit();
              }
            }}
          />

          <button type="submit" className="send-btn" disabled={isSending || (!content.trim() && files.length === 0)}>
            {isSending ? 'Отправка…' : 'Отправить'}
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
