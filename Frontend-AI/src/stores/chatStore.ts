import { create } from 'zustand';
import type { Chat, ChatMessage } from '../types';

interface ChatState {
  chats: Chat[];
  currentChat: Chat | null;
  messages: ChatMessage[];
  isLoadingChats: boolean;
  isLoadingMessages: boolean;
  isSending: boolean;

  // Chats
  setChats: (chats: Chat[]) => void;
  addChat: (chat: Chat) => void;
  updateChat: (chatId: number, updates: Partial<Chat>) => void;
  removeChat: (chatId: number) => void;

  // Current chat
  setCurrentChat: (chat: Chat | null) => void;

  // Messages
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (messageId: number, updates: Partial<ChatMessage>) => void;
  removeMessage: (messageId: number) => void;

  // Loading states
  setLoadingChats: (isLoading: boolean) => void;
  setLoadingMessages: (isLoading: boolean) => void;
  setSending: (isSending: boolean) => void;

  // Clear
  clear: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  chats: [],
  currentChat: null,
  messages: [],
  isLoadingChats: false,
  isLoadingMessages: false,
  isSending: false,

  // Chats
  setChats: (chats) => set({ chats }),

  addChat: (chat) =>
    set((state) => ({
      chats: [chat, ...state.chats],
    })),

  updateChat: (chatId, updates) =>
    set((state) => ({
      chats: state.chats.map((chat) => (chat.id === chatId ? { ...chat, ...updates } : chat)),
      currentChat: state.currentChat?.id === chatId ? { ...state.currentChat, ...updates } : state.currentChat,
    })),

  removeChat: (chatId) =>
    set((state) => ({
      chats: state.chats.filter((chat) => chat.id !== chatId),
      currentChat: state.currentChat?.id === chatId ? null : state.currentChat,
      messages: state.currentChat?.id === chatId ? [] : state.messages,
    })),

  // Current chat
  setCurrentChat: (chat) => set({ currentChat: chat }),

  // Messages
  setMessages: (messages) => set({ messages }),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  updateMessage: (messageId, updates) =>
    set((state) => ({
      messages: state.messages.map((msg) => (msg.id === messageId ? { ...msg, ...updates } : msg)),
    })),

  removeMessage: (messageId) =>
    set((state) => ({
      messages: state.messages.filter((msg) => msg.id !== messageId),
    })),

  // Loading states
  setLoadingChats: (isLoadingChats) => set({ isLoadingChats }),
  setLoadingMessages: (isLoadingMessages) => set({ isLoadingMessages }),
  setSending: (isSending) => set({ isSending }),

  // Clear
  clear: () =>
    set({
      chats: [],
      currentChat: null,
      messages: [],
      isLoadingChats: false,
      isLoadingMessages: false,
      isSending: false,
    }),
}));
