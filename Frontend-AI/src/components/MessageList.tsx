import { useEffect, useRef } from 'react';
import { MessageItem } from './MessageItem';
import { useChatStore } from '../stores/chatStore';
import type { ChatMessage } from '../types';
import './MessageList.css';

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isLoadingMessages } = useChatStore();

  const visibleMessages = messages.filter((m) => !m.is_deleted);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [visibleMessages]);

  if (isLoadingMessages) {
    return (
      <div className="message-list">
        <div className="loading-state">
          <div className="loader" />
          <p>Загружаем историю чата…</p>
        </div>
      </div>
    );
  }

  if (visibleMessages.length === 0) {
    return (
      <div className="message-list">
        <div className="empty-state">
          <p>Сообщений пока нет — задайте первый вопрос ассистенту.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list">
      <div className="messages-container">
        {visibleMessages.map((message) => (
          <MessageItem key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
