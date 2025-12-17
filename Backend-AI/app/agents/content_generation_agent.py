# app/agents/content_generation_agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timezone
from loguru import logger
from contextvars import ContextVar
import json
import io

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.integrations.gigachat_client import gigachat_client, GigaChatClient
from app.integrations import salutespeech_service
from app.integrations.minio_service import MinioService
from app.integrations import minio_service as minio_service_dep
from app.config import settings


# ============================================================================
# CONTEXT
# ============================================================================

@dataclass
class ContentGenContext:
    """Контекст для Content Generation Agent"""
    user_id: int
    default_folder: str = "generated"
    current_pet_name: str = ""


_content_gen_context: ContextVar[Optional[ContentGenContext]] = ContextVar(
    '_content_gen_context',
    default=None
)

_minio_service: ContextVar[Optional[MinioService]] = ContextVar('_minio_service', default=None)


def _get_context() -> ContentGenContext:
    """Get the current context from ContextVar"""
    ctx = _content_gen_context.get()
    if ctx is None:
        raise RuntimeError("ContentGeneration context not set.")
    return ctx


def _get_minio_service() -> MinioService:
    service = _minio_service.get()
    if service is None:
        raise RuntimeError("Minio service not set.")
    return service


# ============================================================================
# TOOLS
# ============================================================================

@tool
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    folder: Optional[str] = None,
) -> str:
    """Сгенерировать изображение через GigaChat и сохранить в MinIO.
    
    Используй для:
    - Создания иллюстраций для статей
    - Визуализации концепций
    - Генерации обучающих материалов
    
    Args:
        prompt: Описание изображения (детальное, на русском)
        width: Ширина изображения (по умолчанию 1024)
        height: Высота изображения (по умолчанию 1024)
        folder: Папка в MinIO (по умолчанию "generated/images")
    
    Returns:
        JSON с информацией о сгенерированном изображении:
        {
          "generated_at": ISO8601,
          "prompt": str,
          "width": int,
          "height": int,
          "minio_object_name": str,
          "minio_url": str,
          "file_size_bytes": int
        }
    """
    try:
        ctx = _get_context()
        minio_service = _get_minio_service()
        
        # Генерируем изображение через GigaChat
        file_id = await gigachat_client.generate_image(
            prompt=prompt,
            width=width,
            height=height
        )
        
        # Скачиваем изображение из GigaChat
        image_bytes = await gigachat_client.download_file(file_id)
        image_io = io.BytesIO(image_bytes)
        
        # Определяем папку
        upload_folder = folder or f"{ctx.default_folder}/images"
        
        # Формируем имя файла
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Сохраняем в MinIO
        minio_object_name = await minio_service.upload_file(
            file=image_io,
            filename=filename,
            content_type="image/png",
            folder=upload_folder
        )
        
        # Получаем URL
        minio_url = await minio_service.get_file_url(minio_object_name)
        
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "width": width,
            "height": height,
            "minio_object_name": minio_object_name,
            "minio_url": minio_url,
            "file_size_bytes": len(image_bytes)
        }
        
        logger.info(f"Image generated and saved: {minio_object_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to generate image: {e}")
        return json.dumps({
            "error": str(e),
            "prompt": prompt
        }, ensure_ascii=False)


@tool
async def create_chart(
    chart_type: Literal["line", "bar", "pie", "scatter", "table"],
    data: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    folder: Optional[str] = None,
) -> str:
    """Создать график, диаграмму или таблицу и сохранить в MinIO.
    
    Используй для:
    - Визуализации данных о здоровье
    - Графиков веса, температуры
    - Диаграмм статистики
    - Таблиц с данными
    
    Args:
        chart_type: Тип графика - line/bar/pie/scatter/table
        data: Данные в JSON формате
              Для line/bar/scatter: {"x": [1,2,3], "y": [4,5,6]} или {"labels": [...], "values": [...]}
              Для pie: {"labels": ["A", "B"], "values": [30, 70]}
              Для table: {"columns": ["Col1", "Col2"], "data": [[1,2], [3,4]]}
        title: Заголовок графика
        x_label: Подпись оси X
        y_label: Подпись оси Y
        folder: Папка в MinIO (по умолчанию "generated/charts")
    
    Returns:
        JSON с информацией о созданном графике:
        {
          "created_at": ISO8601,
          "chart_type": str,
          "title": str,
          "minio_object_name": str,
          "minio_url": str,
          "file_size_bytes": int
        }
    """
    try:
        ctx = _get_context()
        minio_service = _get_minio_service()

        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Парсим данные
        data_dict = json.loads(data)
        
        # Создаём фигуру
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Устанавливаем заголовок
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Создаём график в зависимости от типа
        if chart_type == "line":
            x_data = data_dict.get("x", data_dict.get("labels", []))
            y_data = data_dict.get("y", data_dict.get("values", []))
            ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6)
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "bar":
            x_data = data_dict.get("x", data_dict.get("labels", []))
            y_data = data_dict.get("y", data_dict.get("values", []))
            ax.bar(x_data, y_data, alpha=0.7, color='#4CAF50')
            ax.grid(True, axis='y', alpha=0.3)
            
        elif chart_type == "pie":
            labels = data_dict.get("labels", [])
            values = data_dict.get("values", [])
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            
        elif chart_type == "scatter":
            x_data = data_dict.get("x", [])
            y_data = data_dict.get("y", [])
            ax.scatter(x_data, y_data, alpha=0.6, s=100, color='#FF6B6B')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == "table":
            ax.axis('tight')
            ax.axis('off')
            
            columns = data_dict.get("columns", [])
            table_data = data_dict.get("data", [])
            
            table = ax.table(
                cellText=table_data,
                colLabels=columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.2] * len(columns)
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Стилизация заголовков
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F0F0F0' if row % 2 == 0 else 'white')
        
        # Подписи осей (если не таблица и не круговая)
        if chart_type not in ["table", "pie"]:
            if x_label:
                ax.set_xlabel(x_label, fontsize=11)
            if y_label:
                ax.set_ylabel(y_label, fontsize=11)
        
        # Сохраняем в буфер
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        buffer.seek(0)
        
        # Определяем папку
        upload_folder = folder or f"{ctx.default_folder}/charts"
        
        # Формируем имя файла
        filename = f"chart_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Сохраняем в MinIO
        minio_object_name = await minio_service.upload_file(
            file=buffer,
            filename=filename,
            content_type="image/png",
            folder=upload_folder
        )
        
        # Получаем URL
        minio_url = await minio_service.get_file_url(minio_object_name)
        
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chart_type": chart_type,
            "title": title,
            "minio_object_name": minio_object_name,
            "minio_url": minio_url,
            "file_size_bytes": len(buffer.getvalue())
        }
        
        logger.info(f"Chart created and saved: {minio_object_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to create chart: {e}")
        return json.dumps({
            "error": str(e),
            "chart_type": chart_type
        }, ensure_ascii=False)


@tool
async def text_to_speech(
    text: str,
    voice: str = "Bys_24000",
    audio_format: str = "wav16",
    folder: Optional[str] = None,
) -> str:
    """Синтезировать речь из текста и сохранить в MinIO.
    
    Используй для:
    - Озвучивания текстовых ответов
    - Создания аудио-инструкций
    - Голосовых напоминаний
    
    Args:
        text: Текст для синтеза речи
        voice: Голос (Bys_24000, Nec_24000, May_24000, Ost_24000, Pon_24000)
        audio_format: Формат аудио (wav16, pcm16, opus)
        folder: Папка в MinIO (по умолчанию "generated/audio")
    
    Returns:
        JSON с информацией о синтезированном аудио:
        {
          "synthesized_at": ISO8601,
          "text_preview": str,
          "text_length": int,
          "voice": str,
          "format": str,
          "minio_object_name": str,
          "minio_url": str,
          "file_size_bytes": int
        }
    """
    try:
        ctx = _get_context()
        minio_service = _get_minio_service()
      
        # Синтезируем речь через SaluteSpeech
        audio_bytes = await salutespeech_service.text_to_speech(
            text=text,
            voice=voice,
            format=audio_format
        )
        
        audio_io = io.BytesIO(audio_bytes)
        
        # Определяем папку
        upload_folder = folder or f"{ctx.default_folder}/audio"
        
        # Определяем расширение файла
        extension_map = {
            "wav16": "wav",
            "pcm16": "pcm",
            "opus": "opus"
        }
        ext = extension_map.get(audio_format, "wav")
        
        # Формируем имя файла
        filename = f"tts_{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        
        # Определяем content-type
        content_type_map = {
            "wav16": "audio/wav",
            "pcm16": "audio/pcm",
            "opus": "audio/opus"
        }
        content_type = content_type_map.get(audio_format, "audio/wav")
        
        # Сохраняем в MinIO
        minio_object_name = await minio_service.upload_file(
            file=audio_io,
            filename=filename,
            content_type=content_type,
            folder=upload_folder
        )
        
        # Получаем URL
        minio_url = await minio_service.get_file_url(minio_object_name)
        
        result = {
            "synthesized_at": datetime.now(timezone.utc).isoformat(),
            "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
            "text_length": len(text),
            "voice": voice,
            "format": audio_format,
            "minio_object_name": minio_object_name,
            "minio_url": minio_url,
            "file_size_bytes": len(audio_bytes)
        }
        
        logger.info(f"TTS generated and saved: {minio_object_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to synthesize speech: {e}")
        return json.dumps({
            "error": str(e),
            "text_preview": text[:50]
        }, ensure_ascii=False)


@tool
async def generate_pdf_report(
    title: str,
    content: str,
    folder: Optional[str] = None,
) -> str:
    """Создать PDF отчёт и сохранить в MinIO.
    
    Используй для:
    - Отчётов о здоровье питомца
    - Медицинских справок
    - Сводок по питанию
    
    Args:
        title: Заголовок отчёта
        content: Содержимое отчёта (поддерживает простую разметку: **жирный**)
        folder: Папка в MinIO (по умолчанию "generated/reports")
    
    Returns:
        JSON с информацией о созданном PDF:
        {
          "created_at": ISO8601,
          "title": str,
          "content_length": int,
          "minio_object_name": str,
          "minio_url": str,
          "file_size_bytes": int
        }
    """
    try:
        ctx = _get_context()
        minio_service = _get_minio_service()
    
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Регистрируем шрифт с поддержкой кириллицы
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
            font_name = 'DejaVuSans'
            font_name_bold = 'DejaVuSans-Bold'
        except:
            logger.warning("DejaVu fonts not found, using default")
            font_name = 'Helvetica'
            font_name_bold = 'Helvetica-Bold'
        
        # Создаём PDF в памяти
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # Стили
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name_bold,
            fontSize=18,
            textColor='#2C3E50',
            spaceAfter=20,
            alignment=1  # Center
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=font_name,
            fontSize=11,
            leading=16,
            spaceAfter=12,
        )
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=9,
            textColor='#888888',
        )
        
        # Формируем содержимое
        story = []
        
        # Заголовок
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Дата создания
        date_text = f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        story.append(Paragraph(date_text, footer_style))
        
        # Питомец если указан
        if ctx.current_pet_name:
            pet_text = f"Питомец: {ctx.current_pet_name}"
            story.append(Paragraph(pet_text, footer_style))
        
        story.append(Spacer(1, 0.8*cm))
        
        # Основной контент (разбиваем по параграфам)
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Простая обработка **жирный**
                para_text = para.replace('**', '<b>').replace('**', '</b>')
                story.append(Paragraph(para_text, body_style))
        
        # Генерируем PDF
        doc.build(story)
        
        buffer.seek(0)
        
        # Определяем папку
        upload_folder = folder or f"{ctx.default_folder}/reports"
        
        # Формируем имя файла
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Сохраняем в MinIO
        minio_object_name = await minio_service.upload_file(
            file=buffer,
            filename=filename,
            content_type="application/pdf",
            folder=upload_folder
        )
        
        # Получаем URL
        minio_url = await minio_service.get_file_url(minio_object_name)
        
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "content_length": len(content),
            "minio_object_name": minio_object_name,
            "minio_url": minio_url,
            "file_size_bytes": len(buffer.getvalue())
        }
        
        logger.info(f"PDF report created and saved: {minio_object_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return json.dumps({
            "error": str(e),
            "title": title
        }, ensure_ascii=False)


@tool
async def generate_docx_report(
    title: str,
    content: str,
    folder: Optional[str] = None,
) -> str:
    """Создать DOCX отчёт и сохранить в MinIO.
    
    Используй для:
    - Редактируемых отчётов
    - Документов для печати
    - Шаблонов для заполнения
    
    Args:
        title: Заголовок отчёта
        content: Содержимое отчёта (поддерживает **жирный** текст)
        folder: Папка в MinIO (по умолчанию "generated/reports")
    
    Returns:
        JSON с информацией о созданном DOCX:
        {
          "created_at": ISO8601,
          "title": str,
          "content_length": int,
          "minio_object_name": str,
          "minio_url": str,
          "file_size_bytes": int
        }
    """
    try:
        ctx = _get_context()
        minio_service = _get_minio_service()
        
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        # Создаём документ
        doc = Document()
        
        # Заголовок
        heading = doc.add_heading(title, level=1)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Дата
        date_para = doc.add_paragraph()
        date_run = date_para.add_run(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        date_run.font.size = Pt(10)
        date_run.font.color.rgb = RGBColor(128, 128, 128)
        date_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Питомец если указан
        if ctx.current_pet_name:
            pet_para = doc.add_paragraph()
            pet_run = pet_para.add_run(f"Питомец: {ctx.current_pet_name}")
            pet_run.font.size = Pt(10)
            pet_run.font.color.rgb = RGBColor(100, 100, 100)
            pet_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        doc.add_paragraph()  # Пустая строка
        
        # Основной контент
        paragraphs = content.split('\n\n')
        for para_text in paragraphs:
            if para_text.strip():
                para = doc.add_paragraph()
                
                # Простая обработка **жирный**
                parts = para_text.split('**')
                for i, part in enumerate(parts):
                    if part:
                        run = para.add_run(part)
                        if i % 2 == 1:  # Нечётные части - жирные
                            run.bold = True
                        run.font.size = Pt(11)
        
        # Сохраняем в буфер
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        # Определяем папку
        upload_folder = folder or f"{ctx.default_folder}/reports"
        
        # Формируем имя файла
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        
        # Сохраняем в MinIO
        minio_object_name = await minio_service.upload_file(
            file=buffer,
            filename=filename,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            folder=upload_folder
        )
        
        # Получаем URL
        minio_url = await minio_service.get_file_url(minio_object_name)
        
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "content_length": len(content),
            "minio_object_name": minio_object_name,
            "minio_url": minio_url,
            "file_size_bytes": len(buffer.getvalue())
        }
        
        logger.info(f"DOCX report created and saved: {minio_object_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to generate DOCX report: {e}")
        return json.dumps({
            "error": str(e),
            "title": title
        }, ensure_ascii=False)


# ============================================================================
# CONTENT GENERATION AGENT
# ============================================================================

class ContentGenerationAgent:
    """Агент для генерации контента (изображения, графики, аудио, отчёты)
    
    ВСЕ сгенерированные файлы ВСЕГДА сохраняются в MinIO.
    
    Возможности:
    - Генерация изображений (GigaChat)
    - Создание графиков и таблиц (matplotlib)
    - Синтез речи (SaluteSpeech TTS)
    - Генерация отчётов (PDF, DOCX)
    """
    def __init__(self, minio: Optional[MinioService] = None, llm=None):
        """
        Args:
            minio: Сервис для работы с файлами
            llm: LLM для агента
        """
        from app.integrations.gigachat_client import GigaChatClient
        
        self.minio_service = minio or minio_service_dep
        self.llm = llm or GigaChatClient().llm
        
        # Список инструментов
        self.tools = [
            generate_image,
            create_chart,
            text_to_speech,
            generate_pdf_report,
            generate_docx_report,
        ]
        
        logger.info("ContentGenerationAgent initialized with 5 tools")
    
    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Обработать запрос пользователя"""
        context = context or {}
        token = None
        minio_token = None
        
        try:
            tool_context = ContentGenContext(
                user_id=user_id,
                default_folder="generated",
                current_pet_name=context.get("current_pet_name", "")
            )
            
            # Устанавливаем контексты
            context_token = _content_gen_context.set(tool_context)
            minio_token = _minio_service.set(self.minio_service)
            
            # Информация о питомце
            pet_info = ""
            if tool_context.current_pet_name:
                pet_info = f"\n🐾 Текущий питомец: {tool_context.current_pet_name}"
            
            # System prompt
            system_prompt = f"""Ты - эксперт по генерации контента для владельцев домашних животных.

Пользователь ID: {user_id}{pet_info}

**Доступные инструменты (5):**

1. **generate_image** - Генерация изображений (GigaChat)
   Используй: "Создай картинку", "Нарисуй", "Сгенерируй иллюстрацию"
   
2. **create_chart** - Графики и таблицы (matplotlib)
   Типы: line, bar, pie, scatter, table
   Используй: "Построй график", "Создай диаграмму", "Визуализируй"
   
3. **text_to_speech** - Синтез речи (SaluteSpeech)
   Используй: "Озвучь", "Создай аудио", "Прочитай голосом"
   Голоса: Bys_24000, Nec_24000, May_24000, Ost_24000, Pon_24000
   
4. **generate_pdf_report** - PDF отчёт
   Используй: "Создай PDF", "Сохрани отчёт в PDF"
   
5. **generate_docx_report** - DOCX отчёт
   Используй: "Создай Word документ", "Сделай редактируемый отчёт"

**ВСЕ файлы АВТОМАТИЧЕСКИ сохраняются в MinIO!**

**Форматы данных для графиков:**

Line/Bar/Scatter:
{{{{"labels": ["День 1", "День 2", "День 3"], "values": [10, 15, 12]}}}}

Pie (круговая диаграмма):
{{{{"labels": ["Кошки", "Собаки", "Попугаи"], "values": [30, 50, 20]}}}}

Table (таблица):
{{{{"columns": ["Дата", "Вес (кг)", "Температура"], "data": [["01.12", "5.2", "38.5"], ["02.12", "5.3", "38.3"]]}}}}

**Важно:**
- Каждый инструмент возвращает minio_url для доступа к файлу
- Для generate_image используй детальные русские промпты
- Для графиков добавляй title, x_label, y_label для читаемости
- Можешь указать custom folder для организации файлов

Создавай качественный контент!"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=settings.DEBUG,
                handle_parsing_errors=True,
                max_iterations=5,
            )
            
            result = await agent_executor.ainvoke({"input": user_message})
            return result.get("output", '{"error": "No output"}')
            
        except Exception as e:
            logger.exception(f"ContentGenerationAgent error for user {user_id}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        finally:
            if context_token is not None:
                _content_gen_context.reset(context_token)
            if minio_token is not None:
                _minio_service.reset(minio_token)