import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { ru } from 'date-fns/locale';
import { apiClient, apiErrorMessage } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import { useAuthStore } from '../stores/authStore';
import './Sidebar.css';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const { chatId } = useParams();

  const { user, logout } = useAuthStore();
  const { chats, setChats, setLoadingChats, addChat, updateChat, removeChat } = useChatStore();

  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const token = useMemo(() => localStorage.getItem('access_token'), []);

  const loadChats = useCallback(async () => {
    try {
      setError(null);
      setLoadingChats(true);
      const data = await apiClient.getChats();
      setChats(data);
    } catch (e) {
      setError(apiErrorMessage(e));
      console.error(e);
    } finally {
      setLoadingChats(false);
    }
  }, [setChats, setLoadingChats]);

  useEffect(() => {
    if (!token) return;
    void loadChats();
  }, [token, loadChats]);

  const handleCreateChat = async () => {
    setIsCreating(true);

    try {
      const newChat = await apiClient.createChat({
        title: `Новый диалог ${new Date().toLocaleDateString('ru-RU')}`,
      });

      addChat(newChat);
      navigate(`/chat/${newChat.id}`);
    } catch (error) {
      console.error('Failed to create chat:', error);
      alert(apiErrorMessage(error));
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteChat = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm('Удалить чат и связанные сообщения?')) return;

    try {
      await apiClient.deleteChat(id);
      removeChat(id);

      if (chatId === String(id)) {
        navigate('/');
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
      alert(apiErrorMessage(error));
    }
  };

  const handleRenameChat = async (id: number, currentTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const title = prompt('Новое название чата', currentTitle)?.trim();
    if (!title || title === currentTitle) return;

    try {
      const updated = await apiClient.updateChat(id, { title });
      updateChat(id, updated);
    } catch (error) {
      console.error('Failed to rename chat:', error);
      alert(apiErrorMessage(error));
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <h2>Pet Care</h2>
          <p>Агентный ассистент</p>
        </div>

        <button className="sidebar-close-btn" onClick={onToggle} aria-label="Скрыть список чатов">
          ×
        </button>
      </div>

      <div className="sidebar-user">
        <div className="user-info">
          <span className="user-avatar">{user?.email?.[0]?.toUpperCase() || 'U'}</span>
          <div className="user-details">
            <div className="user-name">{user?.full_name || user?.email || 'Неизвестный пользователь'}</div>
            <div className="user-email">{user?.email}</div>
          </div>
        </div>

        <button className="logout-btn" onClick={logout} title="Выйти">
          ↩︎
        </button>
      </div>

      <button className="new-chat-btn" onClick={handleCreateChat} disabled={isCreating}>
        {isCreating ? 'Создание...' : 'Новый чат'}
      </button>

      {error && <div className="chats-error">{error}</div>}

      <div className="chats-list">
        {chats.length === 0 ? (
          <div className="chats-empty">
            <p>Чатов пока нет</p>
            <small>Создайте новый, чтобы начать диалог и собрать контекст питомца.</small>
          </div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${chatId === String(chat.id) ? 'active' : ''}`}
              onClick={() => navigate(`/chat/${chat.id}`)}
            >
              <div className="chat-item-content">
                <div className="chat-item-title">{chat.title}</div>
                {chat.last_message_at && (
                  <div className="chat-item-time">
                    {formatDistanceToNow(new Date(chat.last_message_at), {
                      addSuffix: true,
                      locale: ru,
                    })}
                  </div>
                )}
              </div>

              <div className="chat-actions">
                <button
                  className="chat-action-btn"
                  onClick={(e) => handleRenameChat(chat.id, chat.title, e)}
                  title="Переименовать чат"
                >
                  ✎
                </button>
                <button
                  className="chat-action-btn danger"
                  onClick={(e) => handleDeleteChat(chat.id, e)}
                  title="Удалить чат"
                >
                  🗑
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
