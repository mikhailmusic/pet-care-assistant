import { useEffect, useState, useRef, type FormEvent } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { apiClient, apiErrorMessage } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import '../styles/AuthPage.css';

export function RegisterPage() {
  const navigate = useNavigate();
  const setUser = useAuthStore((state) => state.setUser);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [googleCredentials, setGoogleCredentials] = useState<string | null>(null);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const exchangeRequestedRef = useRef(false);

  // Используем текущий origin (http://localhost:5173) + путь /register
  const redirectUri = `${window.location.origin}/register`;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');
    const storedState = sessionStorage.getItem('google_oauth_state');

    // Защита от двойного вызова (React Strict Mode + state updates)
    if (!code || exchangeRequestedRef.current) return;

    if (storedState && state && storedState !== state) {
      setError('Не удалось подтвердить Google-авторизацию. Попробуйте снова.');
      sessionStorage.removeItem('google_oauth_state');
      window.history.replaceState({}, '', window.location.pathname);
      return;
    }

    // Устанавливаем флаг ДО асинхронного запроса
    exchangeRequestedRef.current = true;
    setIsGoogleLoading(true);

    void apiClient
      .exchangeGoogleCode(code, redirectUri)
      .then((res) => setGoogleCredentials(res.google_credentials_json))
      .catch((err: unknown) => setError(apiErrorMessage(err)))
      .finally(() => {
        setIsGoogleLoading(false);
        sessionStorage.removeItem('google_oauth_state');
        window.history.replaceState({}, '', window.location.pathname);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [redirectUri]);

  const handleGoogleConnect = async () => {
    if (isGoogleLoading) return;
    try {
      setError('');
      setIsGoogleLoading(true);
      const state =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : Math.random().toString(36).slice(2);
      sessionStorage.setItem('google_oauth_state', state);
      const res = await apiClient.getGoogleAuthUrl(redirectUri, state);
      if (res.state) {
        sessionStorage.setItem('google_oauth_state', res.state);
      }
      window.location.href = res.auth_url;
    } catch (err: unknown) {
      setIsGoogleLoading(false);
      setError(apiErrorMessage(err));
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!googleCredentials) {
      setError('Подключите Google аккаунт, чтобы продолжить.');
      return;
    }

    setIsLoading(true);
    try {
      const response = await apiClient.register({
        email,
        password,
        full_name: fullName || undefined,
        google_credentials_json: googleCredentials,
      });
      setUser(response.user);
      navigate('/');
    } catch (err: any) {
      setError(apiErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1>Pet Care Assistant</h1>
          <p>Зарегистрируйтесь и подключите Google Calendar, чтобы синхронизировать расписание питомца.</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {error && <div className="auth-error">{error}</div>}

          <div className="form-group">
            <label>Google Calendar</label>
            <div className="google-connect">
              <button
                type="button"
                className="auth-button secondary"
                onClick={handleGoogleConnect}
                disabled={isGoogleLoading}
              >
                {isGoogleLoading ? 'Ожидание Google...' : googleCredentials ? 'Подключено' : 'Подключить Google'}
              </button>
              <small>Обязательно: авторизуйтесь в Google и вернитесь, чтобы завершить регистрацию.</small>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="fullName">Имя (необязательно)</label>
            <input
              id="fullName"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="Ваше имя"
              autoComplete="name"
            />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              required
              autoComplete="email"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Пароль</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="********"
              required
              autoComplete="new-password"
              minLength={8}
            />
            <small>Минимум 8 символов.</small>
          </div>

          <button type="submit" className="auth-button" disabled={isLoading || !googleCredentials}>
            {isLoading ? 'Регистрация...' : 'Зарегистрироваться'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Уже есть аккаунт? <Link to="/login">Войти</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
