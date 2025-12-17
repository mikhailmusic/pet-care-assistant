# app/agents/email_agent.py

from __future__ import annotations

from typing import Dict, Any, Optional
from contextvars import ContextVar
from loguru import logger

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
    """
    try:
        service = _get_email_service()
        await service.send_email(to_email=to_email, subject=subject, text=body)
        return f"✅ Письмо отправлено на {to_email} с темой '{subject}'."
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return f"❌ Не удалось отправить письмо: {e}"


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
    ) -> str:
        token = None
        try:
            # Пробрасываем EmailService в tool через ContextVar
            token = _email_service_ctx.set(self.email_service)

            prompt = ChatPromptTemplate.from_messages([
("system", """Ты помощник для отправки email.
Правила:
- Если пользователь просит отправить письмо — подготовь to_email, subject, body и вызови send_email.
- Если email получателя отсутствует — задай ОДИН уточняющий вопрос и НЕ вызывай инструмент.
- Если тема/тело не заданы — сформулируй их по запросу, коротко и по делу.
- Ответ пользователю: либо статус отправки, либо уточняющий вопрос.
- Не используй эмодзи."""),
                ("user", "{input}"),
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
            return result.get("output", "Не удалось обработать запрос на email.")

        except Exception as e:
            logger.exception(f"EmailAgent error for user {user_id}")
            return f"Ошибка при отправке email: {e}"
        finally:
            if token:
                _email_service_ctx.reset(token)
