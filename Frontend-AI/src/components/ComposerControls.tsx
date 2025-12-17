import { useEffect, useState } from 'react';
import { apiClient, apiErrorMessage } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import type { Chat } from '../types';
import './ComposerControls.css';

interface ComposerControlsProps {
  chat: Chat;
}

export function ComposerControls({ chat }: ComposerControlsProps) {
  const { updateChat } = useChatStore();
  const [settings, setSettings] = useState({
    web_search_enabled: chat.web_search_enabled,
    image_generation_enabled: chat.image_generation_enabled,
    voice_response_enabled: chat.voice_response_enabled,
    gigachat_model: chat.gigachat_model,
    message_limit: chat.message_limit,
    temperature: chat.temperature,
    max_tokens: chat.max_tokens ?? 2000,
  });

  useEffect(() => {
    setSettings({
      web_search_enabled: chat.web_search_enabled,
      image_generation_enabled: chat.image_generation_enabled,
      voice_response_enabled: chat.voice_response_enabled,
      gigachat_model: chat.gigachat_model,
      message_limit: chat.message_limit,
      temperature: chat.temperature,
      max_tokens: chat.max_tokens ?? 2000,
    });
  }, [chat]);

  const persist = async (patch: Partial<Chat>) => {
    try {
      const updated = await apiClient.updateChat(chat.id, patch);
      updateChat(chat.id, updated);
    } catch (error) {
      console.error('Failed to update chat settings:', error);
      alert(apiErrorMessage(error));
    }
  };

  return (
    <div className="composer-controls">
      <label className="control toggle">
        <input
          type="checkbox"
          checked={settings.web_search_enabled}
          onChange={(e) => {
            const value = e.target.checked;
            setSettings((s) => ({ ...s, web_search_enabled: value }));
            void persist({ web_search_enabled: value });
          }}
        />
        <span>Веб-поиск</span>
      </label>

      <label className="control toggle">
        <input
          type="checkbox"
          checked={settings.image_generation_enabled}
          onChange={(e) => {
            const value = e.target.checked;
            setSettings((s) => ({ ...s, image_generation_enabled: value }));
            void persist({ image_generation_enabled: value });
          }}
        />
        <span>Генерация изображений</span>
      </label>

      <label className="control toggle">
        <input
          type="checkbox"
          checked={settings.voice_response_enabled}
          onChange={(e) => {
            const value = e.target.checked;
            setSettings((s) => ({ ...s, voice_response_enabled: value }));
            void persist({ voice_response_enabled: value });
          }}
        />
        <span>Голосовой ответ</span>
      </label>

      <div className="advanced-controls">
        <label className="control">
          <span>Модель</span>
          <select
            value={settings.gigachat_model}
            onChange={(e) => {
              const value = e.target.value;
              setSettings((s) => ({ ...s, gigachat_model: value }));
              void persist({ gigachat_model: value });
            }}
          >
            <option value="GigaChat">GigaChat</option>
            <option value="GigaChat-Plus">GigaChat-Plus</option>
            <option value="GigaChat-Pro">GigaChat-Pro</option>
            <option value="GigaChat-Max">GigaChat-Max</option>
          </select>
        </label>

        <label className="control small">
          <span>Лимит сообщений</span>
          <input
            type="number"
            min={1}
            max={100}
            value={settings.message_limit}
            onChange={(e) => {
              const value = parseInt(e.target.value, 10) || 1;
              setSettings((s) => ({ ...s, message_limit: value }));
              void persist({ message_limit: value });
            }}
          />
        </label>

        <label className="control small">
          <span>Температура: {settings.temperature.toFixed(1)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={settings.temperature}
            onChange={(e) => {
              const value = Math.min(1, Math.max(0, parseFloat(e.target.value)));
              setSettings((s) => ({ ...s, temperature: value }));
              void persist({ temperature: value });
            }}
          />
        </label>

        <label className="control small">
          <span>Макс. токенов</span>
          <input
            type="number"
            min={500}
            max={8000}
            step={100}
            value={settings.max_tokens}
            onChange={(e) => {
              const value = parseInt(e.target.value, 10) || 2000;
              setSettings((s) => ({ ...s, max_tokens: value }));
              void persist({ max_tokens: value });
            }}
          />
        </label>
      </div>
    </div>
  );
}
