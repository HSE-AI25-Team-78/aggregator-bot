"""Initial tables

Revision ID: 2bd04e6d67ec
Revises: 
Create Date: 2025-12-23 15:49:44.434061

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision: str = '2bd04e6d67ec'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# In the upgrade() function of 2bd04e6d67ec migration, add existence checks:

def upgrade() -> None:
    """Upgrade schema."""
    # Check if tables exist before creating
    conn = op.get_bind()
    
    # Check if users table exists
    users_exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    ).fetchone()
    
    if not users_exists:
        # Create users table
        op.create_table(
            'users',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('username', sa.String(50), unique=True, nullable=False),
            sa.Column('email', sa.String(100), nullable=True),
            sa.Column('full_name', sa.String(100), nullable=True),
            sa.Column('hashed_password', sa.String(255), nullable=False),
            sa.Column('role', sa.String(20), default='user'),
            sa.Column('is_active', sa.Boolean, default=True),
            sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
            sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
    
    # Check if history table exists
    history_exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
    ).fetchone()
    
    if not history_exists:
        # Create history table
        op.create_table(
            'history',
            sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
            sa.Column('text', sa.Text, nullable=True),
            sa.Column('label', sa.String(50), nullable=True),
            sa.Column('timestamp', sa.DateTime, nullable=False),
            sa.Column('work_time', sa.Float, nullable=False),
            sa.Column('success', sa.Boolean, default=True),
            sa.Column('comment', sa.Text, nullable=True),
            sa.Column('version', sa.String(10), nullable=False)
        )