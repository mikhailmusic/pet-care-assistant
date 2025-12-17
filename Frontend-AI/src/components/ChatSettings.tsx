import { useState, useEffect, useCallback } from 'react';
import { apiClient, apiErrorMessage } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import type { Chat, ChatSettings as ChatSettingsType } from '../types';
import './ChatSettings.css';

interface ChatSettingsProps {
  chat: Chat;
  onClose: () => void;
}

export function ChatSettings({ chat, onClose }: ChatSettingsProps) {
  const { updateChat } = useChatStore();

  const [settings, setSettings] = useState<ChatSettingsType | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  const loadSettings = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await apiClient.getChatSettings(chat.id);
      setSettings(data);
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setIsLoading(false);
    }
  }, [chat.id]);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const handleSave = async () => {
    if (!settings) return;

    setIsSaving(true);

    try {
      const updatedChat = await apiClient.updateChat(chat.id, settings);
      updateChat(chat.id, updatedChat);
      onClose();
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert(apiErrorMessage(error));
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading || !settings) {
    return (
      <div className="chat-settings-overlay" onClick={onClose}>
        <div className="chat-settings-panel" onClick={(e) => e.stopPropagation()}>
          <div className="settings-loading">Загружаем настройки…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-settings-overlay" onClick={onClose}>
      <div className="chat-settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h3>Настройки беседы</h3>
          <button onClick={onClose} className="close-btn" aria-label="Закрыть настройки">
            ×
          </button>
        </div>

        <div className="settings-content">
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.web_search_enabled}
                onChange={(e) => setSettings({ ...settings, web_search_enabled: e.target.checked })}
              />
              <span>Веб-поиск (DuckDuckGo)</span>
            </label>
            <small>Включите, чтобы ассистент мог подтягивать свежие сведения из интернета.</small>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.image_generation_enabled}
                onChange={(e) => setSettings({ ...settings, image_generation_enabled: e.target.checked })}
              />
              <span>Генерация изображений</span>
            </label>
            <small>Использовать GigaChat Image Generation для визуальных ответов.</small>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.voice_response_enabled}
                onChange={(e) => setSettings({ ...settings, voice_response_enabled: e.target.checked })}
              />
              <span>Голосовой ответ</span>
            </label>
            <small>Озвучивать ответы через SaluteSpeech.</small>
          </div>

          <div className="setting-group">
            <label>
              <span>Модель GigaChat</span>
              <select
                value={settings.gigachat_model}
                onChange={(e) => setSettings({ ...settings, gigachat_model: e.target.value })}
              >
                <option value="GigaChat">GigaChat</option>
                <option value="GigaChat-Plus">GigaChat-Plus</option>
                <option value="GigaChat-Pro">GigaChat-Pro</option>
                <option value="GigaChat-Max">GigaChat-Max</option>
              </select>
            </label>
          </div>

          <div className="setting-group">
            <label>
              <span>Лимит сообщений в контексте</span>
              <input
                type="number"
                min="1"
                max="100"
                value={settings.message_limit}
                onChange={(e) => setSettings({ ...settings, message_limit: parseInt(e.target.value, 10) })}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              <span>Температура ({settings.temperature.toFixed(1)})</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.temperature}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    temperature: Math.min(1, Math.max(0, parseFloat(e.target.value))),
                  })
                }
              />
            </label>
            <small>Ниже — более точные ответы, выше — более креативные.</small>
          </div>

          <div className="setting-group">
            <label>
              <span>Ограничение токенов</span>
              <input
                type="number"
                min="500"
                max="8000"
                step="100"
                value={settings.max_tokens ?? 2000}
                onChange={(e) => setSettings({ ...settings, max_tokens: parseInt(e.target.value, 10) })}
              />
            </label>
          </div>
        </div>

        <div className="settings-footer">
          <button onClick={onClose} className="btn-secondary">
            Отмена
          </button>
          <button onClick={handleSave} disabled={isSaving} className="btn-primary">
            {isSaving ? 'Сохраняем…' : 'Сохранить'}
          </button>
        </div>
      </div>
    </div>
  );
}
