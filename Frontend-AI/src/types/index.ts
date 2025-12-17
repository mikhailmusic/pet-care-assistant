// ============================================================================
// API Types - backend DTOs
// ============================================================================

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
  google_credentials_json: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface GoogleAuthUrlResponse {
  auth_url: string;
  state?: string;
}

export interface GoogleCredentialsResponse {
  google_credentials_json: string;
}

// ============================================================================
// Chat Types
// ============================================================================

export interface Chat {
  id: number;
  user_id: number;
  title: string;
  description: string | null;
  web_search_enabled: boolean;
  message_limit: number;
  temperature: number;
  gigachat_model: string;
  image_generation_enabled: boolean;
  voice_response_enabled: boolean;
  max_tokens: number | null;
  created_at: string;
  updated_at: string;
  message_count?: number;
  last_message_at?: string | null;
}

export interface ChatCreateRequest {
  title: string;
  description?: string | null;
}

export interface ChatUpdateRequest {
  title?: string;
  description?: string | null;
  web_search_enabled?: boolean;
  message_limit?: number;
  temperature?: number;
  gigachat_model?: string;
  image_generation_enabled?: boolean;
  voice_response_enabled?: boolean;
  max_tokens?: number | null;
}

export interface ChatSettings {
  web_search_enabled: boolean;
  message_limit: number;
  temperature: number;
  gigachat_model: string;
  image_generation_enabled: boolean;
  voice_response_enabled: boolean;
  max_tokens: number | null;
}

// ============================================================================
// Message Types
// ============================================================================

export type MessageRole = 'user' | 'assistant';
export type MessageType = 'text' | 'image' | 'video' | 'audio' | 'document' | 'mixed';

export interface FileMetadata {
  file_id: string;
  filename: string;
  file_type: 'image' | 'video' | 'audio' | 'document';
  file_size: number;
  mime_type: string;
  url?: string;
}

export interface Message {
  id: number;
  chat_id: number;
  role: MessageRole;
  content: string;
  message_type: MessageType;
  files: FileMetadata[] | null;
  metadata_json: Record<string, any> | null;
  processing_time_ms: number | null;
  created_at: string;
  updated_at: string;
  is_deleted: boolean;
}

export interface MessageCreateRequest {
  content?: string;
  files?: string[] | null;
}

export interface MessageUpdateRequest {
  content?: string;
  files?: string[] | null;
}

// ============================================================================
// Pet Types
// ============================================================================

export interface Pet {
  id: number;
  user_id: number;
  name: string;
  species: string;
  breed: string | null;
  date_of_birth: string | null;
  gender: string | null;
  weight: number | null;
  photo_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface PetCreateRequest {
  name: string;
  species: string;
  breed?: string | null;
  date_of_birth?: string | null;
  gender?: string | null;
  weight?: number | null;
  photo_url?: string | null;
}

// ============================================================================
// Health Record Types
// ============================================================================

export interface HealthRecord {
  id: number;
  pet_id: number;
  record_type: string;
  title: string;
  description: string | null;
  record_date: string;
  veterinarian: string | null;
  clinic: string | null;
  cost: number | null;
  attachments: string[] | null;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// UI State Types
// ============================================================================

export interface UIState {
  isSidebarOpen: boolean;
  isSettingsOpen: boolean;
  currentChatId: number | null;
  isLoading: boolean;
  error: string | null;
}

export interface ChatMessage extends Message {
  isEditing?: boolean;
  isStreaming?: boolean;
}
