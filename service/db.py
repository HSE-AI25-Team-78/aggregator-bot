import os
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import (
    create_engine, Integer, String, Boolean, Text, Float, DateTime, select, delete
)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, Session
from typing import Optional


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default='user')  # 'admin', 'user'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class History(Base):
    __tablename__ = 'history'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    work_time: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(10), nullable=False)


class DataAccessObject:
    def __init__(self, model_version: str = '') -> None:
        self.engine = create_engine('sqlite:///aggregator_bot.db', echo=False)
        Base.metadata.create_all(self.engine)
        self.model_version = model_version
        # self._init_admin_user()  
        
    def _init_admin_user(self):
        """Создает администратора по умолчанию если его нет"""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        with Session(self.engine) as session:
            admin = session.execute(
                select(User).where(User.username == "admin")
            ).scalar_one_or_none()
            
            if not admin:
                admin_user = User(
                    username=os.getenv('ADMIN_USERNAME'),
                    email=os.getenv('ADMIN_EMAIL'),
                    full_name=os.getenv('ADMIN_FULL_NAME'),
                    hashed_password=pwd_context.hash(os.getenv('ADMIN_PASSWORD')),  
                    role="admin",
                    is_active=True
                )
                session.add(admin_user)
                session.commit()
                print("Admin user created")


    def add_history(self, timestamp: datetime, work_time: float,
                    text: str | None = None, label: str | None = None,
                    success: bool = True, comment: str | None = None,
                    version: str | None = None):
        history = History(
            text=text,
            label=label,
            timestamp=timestamp,
            work_time=work_time,
            success=success,
            comment=comment,
            version=version if version is not None else self.model_version
        )
        with Session(self.engine) as session:
            session.add(history)
            session.commit()

    def get_history(self):
        with Session(self.engine) as session:
            select_query = select(
                History.id,
                History.text,
                History.label,
                History.timestamp,
                History.work_time,
                History.success,
                History.comment,
                History.version
            )
            result = session.execute(select_query).mappings().all()
            return result

    def delete_all_history(self) -> bool:
        try:
            with Session(self.engine) as session:
                delete_query = delete(History)
                session.execute(delete_query)
                session.commit()
                return True
        except Exception as e:
            print(f"Error deleting history: {e}")
            return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        with Session(self.engine) as session:
            return session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
    
    def create_user(self, user_data: dict) -> User:
        user = User(**user_data)
        with Session(self.engine) as session:
            session.add(user)
            session.commit()
            session.refresh(user)
        return user
    
    def update_user(self, username: str, update_data: dict) -> Optional[User]:
        with Session(self.engine) as session:
            user = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            
            if user:
                for key, value in update_data.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                session.commit()
                session.refresh(user)
            return user
    
    def delete_user(self, username: str) -> bool:
        with Session(self.engine) as session:
            user = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            
            if user:
                session.delete(user)
                session.commit()
                return True
            return False
    
    def get_all_users(self):
        with Session(self.engine) as session:
            return session.execute(select(User)).scalars().all()
