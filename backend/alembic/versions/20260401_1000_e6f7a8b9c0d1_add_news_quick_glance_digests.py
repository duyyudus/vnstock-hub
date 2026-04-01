"""add news quick glance digests

Revision ID: e6f7a8b9c0d1
Revises: d1e2f3a4b5c6
Create Date: 2026-04-01 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "news_quick_glance_digests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("viewer_key", sa.String(length=32), nullable=False),
        sa.Column("window_hours", sa.Integer(), nullable=False),
        sa.Column("evidence_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "viewer_key",
            "window_hours",
            "evidence_fingerprint",
            name="uq_news_quick_glance_digest_viewer_window_fingerprint",
        ),
    )
    op.create_index(
        "ix_news_quick_glance_digest_viewer_window_generated",
        "news_quick_glance_digests",
        ["viewer_key", "window_hours", "generated_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_news_quick_glance_digests_id"),
        "news_quick_glance_digests",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_news_quick_glance_digests_user_id"),
        "news_quick_glance_digests",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_news_quick_glance_digests_user_id"), table_name="news_quick_glance_digests")
    op.drop_index(op.f("ix_news_quick_glance_digests_id"), table_name="news_quick_glance_digests")
    op.drop_index("ix_news_quick_glance_digest_viewer_window_generated", table_name="news_quick_glance_digests")
    op.drop_table("news_quick_glance_digests")
