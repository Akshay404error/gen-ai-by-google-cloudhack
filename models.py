from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db import Base


class TestCase(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(50), index=True, nullable=True)
    test_title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    preconditions = Column(Text, nullable=True)
    test_steps = Column(Text, nullable=True)  # store as JSON string or newline separated
    test_data = Column(Text, nullable=True)
    expected_result = Column(Text, nullable=True)
    priority = Column(String(20), nullable=True)
    test_type = Column(String(50), nullable=True)
    domain = Column(String(50), nullable=True)
    comments = Column(Text, nullable=True)
    source_story = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)

    __table_args__ = (
        Index("ix_test_cases_domain_priority", "domain", "priority"),
        Index("ix_test_cases_type_created", "test_type", "created_at"),
    )
