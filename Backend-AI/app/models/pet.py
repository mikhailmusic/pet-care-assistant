from sqlalchemy import Column, String, Integer, Float, Date, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from .base import BaseModel


class Pet(BaseModel):    
    __tablename__ = "pets"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic Info (извлекается из диалога)
    name = Column(String(100), nullable=False, index=True)  # "Барсик", "Мурка"
    species = Column(String(50), nullable=False)  # "собака", "кошка", "попугай"
    breed = Column(String(100), nullable=True)  # "лабрадор", "персидская", "волнистый"
    
    # Physical Characteristics
    gender = Column(String(20), nullable=True)  # "самец", "самка"
    color = Column(String(100), nullable=True)  # "рыжий", "черный с белым"
    birth_date = Column(Date, nullable=True)  # Дата рождения
    age_years = Column(Integer, nullable=True)  # Возраст в годах (если точная дата неизвестна)
    age_months = Column(Integer, nullable=True)  # Дополнительные месяцы
    
    # Health Info
    weight_kg = Column(Float, nullable=True)  # Вес в кг
    height_cm = Column(Float, nullable=True)  # Рост/высота в см
    is_sterilized = Column(Boolean, nullable=True)  # True=да, False=нет, None=неизвестно
    microchip_number = Column(String(50), nullable=True)  # Номер чипа
    
    # Medical Info (извлекается из диалога)
    allergies = Column(Text, nullable=True)  
    chronic_conditions = Column(Text, nullable=True)
    medications = Column(Text, nullable=True)
    
    # Lifestyle
    diet_type = Column(String(50), nullable=True)  # "сухой корм", "натуралка", "смешанное"
    activity_level = Column(String(50), nullable=True)  # "низкая", "средняя", "высокая"
    
    # Additional Info
    profile_image_url = Column(String(500), nullable=True)  # Ссылка на фото питомца
    
    # Metadata (для хранения произвольных данных из диалога)
    metadata_json = Column(JSON, nullable=True)
    # Пример: {"last_vaccination": "2024-11-15", "favorite_toy": "мячик", "fears": ["гром", "фейерверки"]}
    
    # Relationships
    user = relationship("User", back_populates="pets")
    health_records = relationship("HealthRecord", back_populates="pet")
    
    def __repr__(self):
        return f"<Pet(id={self.id}, name={self.name}, species={self.species}, breed={self.breed})>"