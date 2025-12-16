import { useEffect, useMemo, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { MessageList } from '../components/MessageList';
import { MessageInput } from '../components/MessageInput';
import { useChatStore } from '../stores/chatStore';
import { apiClient, apiErrorMessage } from '../services/api';
import '../styles/ChatPage.css';

export function ChatPage() {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const { currentChat, messages, setCurrentChat, setMessages, setLoadingMessages, addChat } = useChatStore();

  const currentChatId = currentChat?.id ?? null;
  const hasActiveChat = useMemo(() => !!currentChatId, [currentChatId]);

  const loadChat = useCallback(async (id: number) => {
    try {
      setLoadingMessages(true);
      const [chat, chatMessages] = await Promise.all([apiClient.getChat(id), apiClient.getChatMessages(id)]);

      setCurrentChat(chat);
      setMessages(chatMessages);
    } catch (error) {
      console.error('Failed to load chat:', error);
      navigate('/');
    } finally {
      setLoadingMessages(false);
    }
  }, [navigate, setCurrentChat, setMessages, setLoadingMessages]);

  useEffect(() => {
    if (chatId) {
      void loadChat(parseInt(chatId, 10));
    } else {
      setCurrentChat(null);
      setMessages([]);
    }
  }, [chatId, loadChat, setCurrentChat, setMessages]);

  const handleStartChat = async () => {
    try {
      setLoadingMessages(true);
      const newChat = await apiClient.createChat({
        title: `Новый диалог ${new Date().toLocaleDateString('ru-RU')}`,
      });

      addChat(newChat);
      setCurrentChat(newChat);
      setMessages([]);
      navigate(`/chat/${newChat.id}`);
    } catch (error) {
      console.error('Failed to start chat:', error);
      alert(apiErrorMessage(error));
    } finally {
      setLoadingMessages(false);
    }
  };

  return (
    <div className="chat-page">
      <Sidebar isOpen={isSidebarOpen} onToggle={() => setIsSidebarOpen(!isSidebarOpen)} />

      <div className="chat-main">
        <div className="chat-header">
          <div className="chat-header-left">
            {!isSidebarOpen && (
              <button className="sidebar-toggle-btn" onClick={() => setIsSidebarOpen(true)} aria-label="Открыть список чатов">
                ☰
              </button>
            )}

            {currentChat && (
              <div className="chat-title">
                <h2>{currentChat.title}</h2>
                {currentChat.description && <p className="chat-description">{currentChat.description}</p>}
              </div>
            )}
          </div>

        </div>

        <div className="chat-content">
          {!hasActiveChat ? (
            <div className="chat-empty-state">
              <h1>Pet Care Assistant</h1>
              <p>Создайте новый чат, чтобы начать диалог, загрузить файлы и собрать данные о питомце.</p>
              <button className="primary-btn" onClick={handleStartChat}>
                Новый чат
              </button>
            </div>
          ) : (
            currentChatId && (
              <>
                <MessageList messages={messages} />
                <MessageInput chatId={currentChatId} />
              </>
            )
          )}
        </div>
      </div>

    </div>
  );
}
