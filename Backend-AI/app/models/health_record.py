from sqlalchemy import Column, String, Integer, Float, Date, ForeignKey, Text, JSON, Enum, Boolean
from sqlalchemy.orm import relationship
from .base import BaseModel
import enum


class RecordType(str, enum.Enum):
    VACCINATION = "vaccination"  # Прививка
    EXAMINATION = "examination"  # Осмотр
    TREATMENT = "treatment"  # Лечение
    SURGERY = "surgery"  # Операция
    ANALYSIS = "analysis"  # Анализы
    SYMPTOM = "symptom"  # Симптом/жалоба
    MEDICATION = "medication"  # Прием лекарств
    WEIGHT = "weight"  # Взвешивание
    BEHAVIOR = "behavior"  # Поведенческие наблюдения
    OTHER = "other"  # Прочее


class UrgencyLevel(str, enum.Enum):
    LOW = "low"  # Низкая (плановое)
    MEDIUM = "medium"  # Средняя (наблюдение)
    HIGH = "high"  # Высокая (требуется визит к ветеринару)
    CRITICAL = "critical"  # Критическая (срочно!)


class HealthRecord(BaseModel):  
    __tablename__ = "health_records"
    
    pet_id = Column(Integer, ForeignKey("pets.id"), nullable=False, index=True)
    

    record_type = Column(Enum(RecordType), nullable=False, index=True)
    record_date = Column(Date, nullable=False, index=True)  # Дата события
    
    title = Column(String(255), nullable=False)  # "Прививка от бешенства", "Диарея"
    description = Column(Text, nullable=True)  # Подробное описание
    
    symptoms = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=True)  # Диагноз
    treatment = Column(Text, nullable=True)  # Назначенное лечение
    medications_prescribed = Column(String(255), nullable=True)  # Прописанные лекарства
    
    weight_kg = Column(Float, nullable=True)  # Вес на момент записи
    temperature_c = Column(Float, nullable=True)  # Температура
    
    urgency = Column(Enum(UrgencyLevel), default=UrgencyLevel.LOW, nullable=False)
    is_resolved = Column(Boolean, nullable=False)  # True/False/None
    
    vet_name = Column(String(255), nullable=True)  # Имя ветеринара
    vet_clinic = Column(String(255), nullable=True)  # Клиника
    
    files = Column(JSON, nullable=True)  # Ссылки на файлы (анализы, рентген и т.д.)
    # Пример: [{"type": "pdf", "name": "анализы.pdf", "url": "minio://..."}, {"type": "image", "name": "рентген.jpg", "url": "..."}]
    
    cost = Column(Float, nullable=True)  # Стоимость (если применимо)
    
    next_visit_date = Column(Date, nullable=True)  # Дата следующего визита
    
    metadata_json = Column(JSON, nullable=True)  # Дополнительные данные
    
    pet = relationship("Pet", back_populates="health_records")
    
    def __repr__(self):
        return f"<HealthRecord(id={self.id}, pet_id={self.pet_id}, type={self.record_type}, title={self.title})>"