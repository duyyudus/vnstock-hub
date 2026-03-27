"""add news tables

Revision ID: c9d8e7f6a5b4
Revises: f4a5b6c7d8e9
Create Date: 2026-03-26 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c9d8e7f6a5b4"
down_revision: Union[str, None] = "f4a5b6c7d8e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "news_sites",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("homepage_url", sa.String(length=1000), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("domain", "homepage_url", name="uq_news_site_domain_homepage"),
    )
    op.create_index("ix_news_sites_domain", "news_sites", ["domain"])
    op.create_index("ix_news_sites_public", "news_sites", ["is_public"])

    op.create_table(
        "news_site_feeds",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("site_id", sa.Integer(), nullable=False),
        sa.Column("feed_url", sa.String(length=1000), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="rss"),
        sa.Column("discovery_method", sa.String(length=20), nullable=False, server_default="manual"),
        sa.Column("validation_status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("validation_error", sa.String(length=1000), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("poll_interval_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("last_polled_at", sa.DateTime(), nullable=True),
        sa.Column("next_poll_at", sa.DateTime(), nullable=True),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["news_sites.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("feed_url", name="uq_news_site_feeds_feed_url"),
    )
    op.create_index("ix_news_site_feeds_status_next_poll", "news_site_feeds", ["validation_status", "next_poll_at"])

    op.create_table(
        "news_crawl_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("site_id", sa.Integer(), nullable=False),
        sa.Column("listing_url", sa.String(length=1000), nullable=False),
        sa.Column("article_link_selector", sa.String(length=255), nullable=False),
        sa.Column("content_selector", sa.String(length=255), nullable=False),
        sa.Column("excerpt_selector", sa.String(length=255), nullable=True),
        sa.Column("pagination_config", sa.JSON(), nullable=True),
        sa.Column("validation_status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("validation_error", sa.String(length=1000), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("poll_interval_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("last_polled_at", sa.DateTime(), nullable=True),
        sa.Column("next_poll_at", sa.DateTime(), nullable=True),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["news_sites.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("listing_url", "article_link_selector", name="uq_news_crawl_source_listing_selector"),
    )
    op.create_index("ix_news_crawl_sources_status_next_poll", "news_crawl_sources", ["validation_status", "next_poll_at"])

    op.create_table(
        "news_articles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("canonical_url", sa.String(length=1000), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("language", sa.String(length=20), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("image_url", sa.String(length=1000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("canonical_url", name="uq_news_articles_canonical_url"),
    )
    op.create_index("ix_news_articles_published_at", "news_articles", ["published_at"])
    op.create_index("ix_news_articles_content_hash", "news_articles", ["content_hash"])

    op.create_table(
        "news_source_subscriptions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("site_feed_id", sa.Integer(), nullable=True),
        sa.Column("crawl_source_id", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["crawl_source_id"], ["news_crawl_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["site_feed_id"], ["news_site_feeds.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "crawl_source_id", name="uq_news_subscription_user_crawl"),
        sa.UniqueConstraint("user_id", "site_feed_id", name="uq_news_subscription_user_feed"),
    )
    op.create_index("ix_news_source_subscriptions_user_enabled", "news_source_subscriptions", ["user_id", "enabled"])

    op.create_table(
        "news_article_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("site_feed_id", sa.Integer(), nullable=True),
        sa.Column("crawl_source_id", sa.Integer(), nullable=True),
        sa.Column("article_url", sa.String(length=1000), nullable=False),
        sa.Column("discovered_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["news_articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["crawl_source_id"], ["news_crawl_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["site_feed_id"], ["news_site_feeds.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("article_id", "crawl_source_id", name="uq_news_article_source_article_crawl"),
        sa.UniqueConstraint("article_id", "site_feed_id", name="uq_news_article_source_article_feed"),
    )

    op.create_table(
        "news_article_semantics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("topics", sa.JSON(), nullable=False),
        sa.Column("tickers", sa.JSON(), nullable=False),
        sa.Column("sectors", sa.JSON(), nullable=False),
        sa.Column("importance", sa.String(length=20), nullable=True),
        sa.Column("sentiment", sa.String(length=20), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("classified_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["news_articles.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("article_id", name="uq_news_article_semantics_article_id"),
    )
    op.create_index("ix_news_article_semantics_importance", "news_article_semantics", ["importance"])
    op.create_index("ix_news_article_semantics_sentiment", "news_article_semantics", ["sentiment"])

    op.create_table(
        "news_user_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("blocked_topics_text", sa.Text(), nullable=True),
        sa.Column("blocked_labels", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", name="uq_news_user_preferences_user_id"),
    )

    op.create_table(
        "news_ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(length=20), nullable=False),
        sa.Column("site_feed_id", sa.Integer(), nullable=True),
        sa.Column("crawl_source_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("fetched_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stored_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("filtered_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["crawl_source_id"], ["news_crawl_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["site_feed_id"], ["news_site_feeds.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_news_ingestion_runs_status_started_at", "news_ingestion_runs", ["status", "started_at"])


def downgrade() -> None:
    op.drop_index("ix_news_ingestion_runs_status_started_at", table_name="news_ingestion_runs")
    op.drop_table("news_ingestion_runs")

    op.drop_table("news_user_preferences")

    op.drop_index("ix_news_article_semantics_sentiment", table_name="news_article_semantics")
    op.drop_index("ix_news_article_semantics_importance", table_name="news_article_semantics")
    op.drop_table("news_article_semantics")

    op.drop_table("news_article_sources")

    op.drop_index("ix_news_source_subscriptions_user_enabled", table_name="news_source_subscriptions")
    op.drop_table("news_source_subscriptions")

    op.drop_index("ix_news_articles_content_hash", table_name="news_articles")
    op.drop_index("ix_news_articles_published_at", table_name="news_articles")
    op.drop_table("news_articles")

    op.drop_index("ix_news_crawl_sources_status_next_poll", table_name="news_crawl_sources")
    op.drop_table("news_crawl_sources")

    op.drop_index("ix_news_site_feeds_status_next_poll", table_name="news_site_feeds")
    op.drop_table("news_site_feeds")

    op.drop_index("ix_news_sites_public", table_name="news_sites")
    op.drop_index("ix_news_sites_domain", table_name="news_sites")
    op.drop_table("news_sites")
