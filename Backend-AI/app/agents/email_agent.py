# app/agents/email_agent.py

from __future__ import annotations

from typing import Dict, Any, Optional, List
from contextvars import ContextVar
from loguru import logger
import json

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.integrations.email_service import EmailService
from app.integrations.gigachat_client import GigaChatClient
from app.config import settings


_email_service_ctx: ContextVar[Optional[EmailService]] = ContextVar("_email_service_ctx", default=None)


def _get_email_service() -> EmailService:
    service = _email_service_ctx.get()
    if service is None:
        raise RuntimeError("Email service not set")
    return service


@tool
async def send_email(to_email: str, subject: str, body: str) -> str:
    """
    Отправить письмо по email.

    Args:
        to_email: Адрес получателя.
        subject: Тема письма (до ~100 символов).
        body: Текст письма в свободной форме.

    Returns:
        JSON с информацией об отправленном письме:
        {
          "email_sent": true,
          "recipient_email": str,
          "subject": str,
          "body_preview": str
        }
    """
    try:
        service = _get_email_service()
        await service.send_email(to_email=to_email, subject=subject, text=body)

        result = {
            "email_sent": True,
            "recipient_email": to_email,
            "subject": subject,
            "body_preview": body[:100] + ("..." if len(body) > 100 else "")
        }

        logger.info(f"Email sent to {to_email}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return json.dumps({
            "email_sent": False,
            "error": str(e),
            "recipient_email": to_email
        }, ensure_ascii=False)


class EmailAgent:
    """Агент для подготовки и отправки писем через SMTP."""

    def __init__(self, email_service: EmailService, llm=None):
        self.email_service = email_service
        self.llm = llm or GigaChatClient().llm
        self.tools = [send_email]
        logger.info("EmailAgent initialized with send_email tool")

    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        token = None
        try:
            # Пробрасываем EmailService в tool через ContextVar
            token = _email_service_ctx.set(self.email_service)

            # Формируем контекст истории разговора для промпта
            history_context = ""
            if conversation_history:
                history_items = []
                for msg in conversation_history[-10:]:  # Берём последние 10 сообщений для контекста
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history_items.append(f"Пользователь: {content}")
                    elif role == "assistant":
                        history_items.append(f"Ассистент: {content}")

                if history_items:
                    history_context = "\n\nИСТОРИЯ РАЗГОВОРА (для справки):\n" + "\n".join(history_items)

            prompt = ChatPromptTemplate.from_messages([
("system", f"""Ты помощник для отправки email.
Правила:
- Если пользователь просит отправить письмо — подготовь to_email, subject, body и вызови send_email.
- Если email получателя отсутствует — задай ОДИН уточняющий вопрос и НЕ вызывай инструмент.
- Если тема/тело не заданы явно — сформулируй их на основе запроса пользователя.
- ВАЖНО: Если пользователь ссылается на информацию из предыдущих сообщений ("информацию, которую ты советовал", "то, что мы обсуждали"), используй историю разговора ниже для формирования body письма.

**КОНТЕКСТ:**
Если пользователь просит отправить "информацию о [чём-то]" или "то, что ты советовал", найди эту информацию в истории разговора и включи её ПОЛНОСТЬЮ в body письма.{history_context}

**КРИТИЧЕСКИ ВАЖНО:**
После вызова инструмента send_email:
- Инструмент возвращает валидный JSON с полями: email_sent, recipient_email, subject, body_preview
- Верни ТОЧНО ТАКОЙ ЖЕ JSON без изменений
- Используй ТОЛЬКО стандартные двойные кавычки " (не используй специальные токены)
- НЕ добавляй свой текст до или после JSON
- НЕ упрощай JSON - верни ВСЕ поля как есть
- НЕ изменяй структуру JSON
- Формат ответа: чистый JSON без markdown обёрток

Пример правильного ответа:
{{{{
  "email_sent": true,
  "recipient_email": "user@example.com",
  "subject": "Тема письма",
  "body_preview": "Текст письма..."
}}}}"""),
                ("user", "{{input}}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=settings.DEBUG,
                handle_parsing_errors=True,
                max_iterations=3,
            )

            result = await executor.ainvoke({"input": user_message})
            output = result.get("output", "Не удалось обработать запрос на email.")

            logger.info(f"EmailAgent raw output: {output[:500]}")

            # Пытаемся получить оригинальный вывод инструмента из intermediate_steps
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                last_action, last_output = intermediate_steps[-1]
                tool_name = getattr(last_action, 'tool', None)

                if tool_name == 'send_email':
                    logger.info(f"Using original tool output from intermediate_steps")
                    try:
                        if isinstance(last_output, str):
                            parsed = json.loads(last_output)
                            if "email_sent" in parsed:
                                logger.info(f"Returning validated email JSON output directly")
                                return json.dumps(parsed, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        logger.warning(f"Tool output is not valid JSON")

                    output = last_output

            # Пытаемся извлечь чистый JSON из ответа
            try:
                if "{" in output and "}" in output:
                    # Убираем двойные фигурные скобки если есть ({{ -> {)
                    cleaned_output = output.replace("{{ ", "{ ").replace(" }}", " }")

                    # Убираем служебные токены GigaChat
                    cleaned_output = cleaned_output.replace("<|superquote|>", '"')

                    start_idx = cleaned_output.find("{")
                    end_idx = cleaned_output.rfind("}") + 1
                    potential_json = cleaned_output[start_idx:end_idx]

                    # Пытаемся распарсить
                    try:
                        parsed = json.loads(potential_json)
                    except json.JSONDecodeError as e:
                        # Если не получилось из-за control characters, экранируем переносы строк
                        import re
                        # Ищем строковые значения и экранируем в них переносы строк
                        # Заменяем буквальные \n на экранированные \\n внутри строк
                        def escape_newlines_in_strings(match):
                            value = match.group(1)
                            # Экранируем переносы строк
                            value = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                            return f'"{value}"'

                        # Паттерн для поиска строковых значений в JSON
                        potential_json = re.sub(r'"([^"]*)"', escape_newlines_in_strings, potential_json, flags=re.DOTALL)
                        parsed = json.loads(potential_json)

                    if "email_sent" in parsed:
                        clean_json = json.dumps(parsed, ensure_ascii=False, indent=2)
                        logger.info(f"Extracted clean JSON from agent output")
                        return clean_json
            except Exception as e:
                logger.warning(f"Failed to extract JSON from output: {e}")

            # Если не удалось извлечь JSON, оборачиваем в JSON
            if not output.strip().startswith("{"):
                logger.warning(f"Output is not JSON, wrapping in response")
                return json.dumps({
                    "email_sent": False,
                    "message": output
                }, ensure_ascii=False, indent=2)

            return output

        except Exception as e:
            logger.exception(f"EmailAgent error for user {user_id}")
            return json.dumps({
                "email_sent": False,
                "error": str(e)
            }, ensure_ascii=False)
        finally:
            if token:
                _email_service_ctx.reset(token)
