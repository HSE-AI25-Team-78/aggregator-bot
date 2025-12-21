from datetime import datetime
from sqlalchemy import (
    create_engine, Integer, String, Boolean, Text, Float, DateTime, select
)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, Session
from typing import Optional


Base = declarative_base()


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


