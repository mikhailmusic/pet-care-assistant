import axios, { type AxiosInstance, type AxiosError } from 'axios';
import type {
  AuthResponse,
  LoginRequest,
  RegisterRequest,
  User,
  GoogleAuthUrlResponse,
  GoogleCredentialsResponse,
  Chat,
  ChatCreateRequest,
  ChatUpdateRequest,
  ChatSettings,
  Message,
  MessageCreateRequest,
  MessageUpdateRequest,
  Pet,
  PetCreateRequest,
  HealthRecord,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          localStorage.removeItem('user');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth
  async login(data: LoginRequest): Promise<AuthResponse> {
    const formData = new FormData();
    formData.append('username', data.email);
    formData.append('password', data.password);

    const response = await this.client.post<AuthResponse>('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    localStorage.setItem('access_token', response.data.access_token);
    localStorage.setItem('user', JSON.stringify(response.data.user));

    return response.data;
  }

  async register(data: RegisterRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data);
    localStorage.setItem('access_token', response.data.access_token);
    localStorage.setItem('user', JSON.stringify(response.data.user));
    return response.data;
  }

  async getGoogleAuthUrl(redirectUri: string, state?: string): Promise<GoogleAuthUrlResponse> {
    const response = await this.client.get<GoogleAuthUrlResponse>('/auth/google/url', {
      params: { redirect_uri: redirectUri, state },
    });
    return response.data;
  }

  async exchangeGoogleCode(code: string, redirectUri: string): Promise<GoogleCredentialsResponse> {
    const response = await this.client.post<GoogleCredentialsResponse>('/auth/google/exchange', {
      code,
      redirect_uri: redirectUri,
    });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }

  // Chats
  async getChats(skip = 0, limit = 50): Promise<Chat[]> {
    const response = await this.client.get<Chat[]>('/chats', { params: { skip, limit } });
    return response.data;
  }

  async getChat(chatId: number, withMessages = false): Promise<Chat> {
    const response = await this.client.get<Chat>(`/chats/${chatId}`, { params: { with_messages: withMessages } });
    return response.data;
  }

  async createChat(data: ChatCreateRequest): Promise<Chat> {
    const response = await this.client.post<Chat>('/chats', data);
    return response.data;
  }

  async updateChat(chatId: number, data: ChatUpdateRequest): Promise<Chat> {
    const response = await this.client.patch<Chat>(`/chats/${chatId}`, data);
    return response.data;
  }

  async deleteChat(chatId: number): Promise<void> {
    await this.client.delete(`/chats/${chatId}`);
  }

  async getChatSettings(chatId: number): Promise<ChatSettings> {
    const response = await this.client.get<ChatSettings>(`/chats/${chatId}/settings`);
    return response.data;
  }

  // Messages
  async getChatMessages(chatId: number, skip = 0, limit = 100, order = 'asc'): Promise<Message[]> {
    const response = await this.client.get<Message[]>(`/chats/${chatId}/messages`, {
      params: { skip, limit, order },
    });
    return response.data;
  }

  async createMessage(chatId: number, data: MessageCreateRequest): Promise<Message> {
    const response = await this.client.post<Message>(`/chats/${chatId}/messages`, data);
    return response.data;
  }

  // запуск пайплайна агентов и возврат ответа ассистента
  async sendMessage(chatId: number, data: MessageCreateRequest): Promise<Message> {
    const response = await this.client.post<Message>(`/chats/${chatId}/send`, data);
    return response.data;
  }

  // обновление пользовательского сообщения и перезапуск цепочки ассистента
  async updateMessage(messageId: number, data: MessageUpdateRequest): Promise<Message> {
    const response = await this.client.patch<Message>(`/messages/${messageId}`, data);
    return response.data;
  }

  async deleteMessage(messageId: number): Promise<void> {
    await this.client.delete(`/messages/${messageId}`);
  }

  // Pets
  async getPets(): Promise<Pet[]> {
    const response = await this.client.get<Pet[]>('/pets');
    return response.data;
  }

  async getPet(petId: number): Promise<Pet> {
    const response = await this.client.get<Pet>(`/pets/${petId}`);
    return response.data;
  }

  async createPet(data: PetCreateRequest): Promise<Pet> {
    const response = await this.client.post<Pet>('/pets', data);
    return response.data;
  }

  async updatePet(petId: number, data: Partial<PetCreateRequest>): Promise<Pet> {
    const response = await this.client.patch<Pet>(`/pets/${petId}`, data);
    return response.data;
  }

  async deletePet(petId: number): Promise<void> {
    await this.client.delete(`/pets/${petId}`);
  }

  // Health Records
  async getHealthRecords(petId: number): Promise<HealthRecord[]> {
    const response = await this.client.get<HealthRecord[]>(`/pets/${petId}/health-records`);
    return response.data;
  }

  // Files
  async uploadFile(file: File): Promise<{ file_id: string; url: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<{ file_id: string; url: string }>('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return response.data;
  }

  async deleteFile(fileId: string): Promise<void> {
    await this.client.delete(`/files/${fileId}`);
  }
}

export function apiErrorMessage(err: unknown): string {
  if (!axios.isAxiosError(err)) return 'Неизвестная ошибка';

  const status = err.response?.status;
  const url = (err.config?.baseURL || '') + (err.config?.url || '');

  if (status === 404) return `API endpoint не найден (404): ${url}`;
  if (status === 401) return 'Сессия устарела. Войдите заново.';
  if (status === 403) return 'Недостаточно прав для действия.';
  if (status === 422) {
    const detail: any = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((x) => x?.msg).filter(Boolean).join(', ');
    return 'Некорректные данные (422)';
  }

  if (!err.response) return `Нет соединения с API: ${err.message}`;

  return (err.response?.data as any)?.detail || `Ошибка API (${status})`;
}

export const apiClient = new APIClient();
