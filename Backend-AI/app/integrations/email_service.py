from __future__ import annotations

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from loguru import logger

from app.config import settings
from app.utils.exceptions import EmailSendException


class EmailService:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: Optional[bool] = None,
    ) -> None:

        self.host = host or settings.SMTP_HOST
        self.port = port or settings.SMTP_PORT
        self.username = username or settings.SMTP_USERNAME
        self.password = password or settings.SMTP_PASSWORD
        self.use_tls = settings.SMTP_USE_TLS if use_tls is None else use_tls

        self.from_email = self.username

        logger.info(
            "EmailService initialized: host=%s port=%s TLS=%s sender=%s",
            self.host, self.port, self.use_tls, self.from_email
        )

    async def send_email(self, to_email: str,subject: str,text: str,) -> None:

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = to_email
        msg.attach(MIMEText(text, 'plain', 'utf-8'))

        def _sync_send():
            try:
                logger.info("Sending email to %s (subject='%s')", to_email, subject)

                server = smtplib.SMTP(self.host, self.port)
                server.starttls()
                
                server.login(self.username, self.password)

                server.send_message(msg)
                server.quit()

                logger.info("Email successfully sent to %s", to_email)

            except Exception as e:
                logger.error(f"SMTP error: {e}")
                raise EmailSendException(str(e))

        await asyncio.to_thread(_sync_send)


email_service = EmailService()