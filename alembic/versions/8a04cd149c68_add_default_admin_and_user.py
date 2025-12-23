"""Add default admin and user

Revision ID: 8a04cd149c68
Revises: 2bd04e6d67ec
Create Date: 2025-12-23 15:50:21.222745

"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()

# revision identifiers, used by Alembic.
revision: str = '8a04cd149c68'
down_revision: Union[str, Sequence[str], None] = '2bd04e6d67ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Insert default admin user
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    current_time = datetime.now()
    
    # Get database connection
    conn = op.get_bind()
    
    # Check if users already exist - using fetchone()[0] instead of scalar()
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM users WHERE username IN (:user1, :user2)"),
        {"user1": "admin", "user2": "default_user"}
    ).fetchone()
    
    if result and result[0] == 0:  # Only insert if users don't exist
        op.bulk_insert(
            sa.table('users',
                sa.column('username', sa.String),
                sa.column('email', sa.String),
                sa.column('full_name', sa.String),
                sa.column('hashed_password', sa.String),
                sa.column('role', sa.String),
                sa.column('is_active', sa.Boolean),
                sa.column('created_at', sa.DateTime),
                sa.column('updated_at', sa.DateTime)
            ),
            [
                {
                    'username': os.getenv('ADMIN_USERNAME', 'admin'),
                    'email': os.getenv('ADMIN_EMAIL', 'admin@example.com'),
                    'full_name': os.getenv('ADMIN_FULL_NAME', 'System Administrator'),
                    'hashed_password': pwd_context.hash(os.getenv('ADMIN_PASSWORD', 'admin123')),
                    'role': 'admin',
                    'is_active': True,
                    'created_at': current_time,
                    'updated_at': current_time
                },
                {
                    'username': 'default_user',
                    'email': 'user@example.com',
                    'full_name': 'Default User',
                    'hashed_password': pwd_context.hash('user123'),
                    'role': 'user',
                    'is_active': True,
                    'created_at': current_time,
                    'updated_at': current_time
                }
            ]
        )
        print("Default users added successfully")
    else:
        print(f"Users already exist (count: {result[0] if result else 'unknown'}), skipping...")


def downgrade() -> None:
    """Downgrade schema."""
    # Remove the default users
    op.execute("DELETE FROM users WHERE username IN ('admin', 'default_user')")