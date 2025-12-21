from __future__ import annotations

from typing import Optional, Annotated
from loguru import logger
import json

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.integrations.email_service import EmailService


class EmailTools:
    
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
    
    @tool
    async def send_email(
        self,
        state: Annotated[dict, InjectedState],
        to_email: str,
        subject: str,
        body: str,
    ) -> str:
        """Отправить письмо по email.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            to_email: Адрес получателя
            subject: Тема письма (до ~100 символов)
            body: Текст письма в свободной форме
        
        Returns:
            JSON с информацией об отправленном письме
        """
        try:
            await self.email_service.send_email(
                to_email=to_email,
                subject=subject,
                text=body
            )
            
            result = {
                "email_sent": True,
                "recipient_email": to_email,
                "subject": subject,
                "body_preview": body[:500] + ("..." if len(body) > 500 else "")
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



def create_email_agent(
    email_service: EmailService,
    llm,
    name: str = "email",
):
    """Создать агента для отправки email
    
    Args:
        email_service: Сервис для отправки email через SMTP
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = EmailTools(email_service)
    
    tools = [
        tools_instance.send_email,
    ]
    
    prompt = (
        "Ты - помощник для отправки email.\n\n"
        "Твои возможности:\n"
        "- Отправка писем на указанный email адрес\n\n"
        "Правила:\n"
        "- Если пользователь просит отправить письмо - подготовь to_email, subject, body и вызови send_email\n"
        "- Если email получателя отсутствует - попроси пользователя указать его\n"
        "- Если тема/тело не заданы явно - сформулируй их на основе контекста разговора\n"
        "- Если пользователь ссылается на информацию из предыдущих сообщений - используй историю сообщений из state\n\n"
        "Будь вежливым и точным!"
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created EmailAgent '{name}' with {len(tools)} tools")
    return agent
