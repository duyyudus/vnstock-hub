from sqlalchemy import Column, String, Integer, Float, BigInteger, Date, DateTime, UniqueConstraint, Index, JSON, Boolean, ForeignKey, Text
from datetime import datetime
from app.db.database import Base

class StockCompany(Base):
    """Model to store company full names for stock symbols."""
    __tablename__ = "stock_companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
    exchange = Column(String(20), nullable=True)
    charter_capital = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockIndex(Base):
    """Model to store stock market indices."""
    __tablename__ = "stock_indices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=True)
    group = Column(String(100), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockDailyHistory(Base):
    """Canonical daily stock history with OHLCV and optional flow enrichment."""
    __tablename__ = "stock_daily_history"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger)
    matched_volume = Column(BigInteger, nullable=True)
    matched_value = Column(Float, nullable=True)
    deal_volume = Column(BigInteger, nullable=True)
    deal_value = Column(Float, nullable=True)
    total_volume = Column(BigInteger, nullable=True)
    total_value = Column(Float, nullable=True)
    foreign_buy_volume = Column(BigInteger, nullable=True)
    foreign_buy_value = Column(Float, nullable=True)
    foreign_sell_volume = Column(BigInteger, nullable=True)
    foreign_sell_value = Column(Float, nullable=True)
    foreign_net_volume = Column(BigInteger, nullable=True)
    foreign_net_value = Column(Float, nullable=True)
    prop_buy_volume = Column(BigInteger, nullable=True)
    prop_buy_value = Column(Float, nullable=True)
    prop_sell_volume = Column(BigInteger, nullable=True)
    prop_sell_value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_daily_history_symbol_date'),
        Index('ix_stock_daily_history_symbol_date', 'symbol', 'date'),
    )


class StockHistorySyncState(Base):
    """State tracking for deterministic history full/incremental sync."""
    __tablename__ = "stock_history_sync_state"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    listing_date = Column(Date, nullable=True)
    sync_status = Column(String(20), nullable=False, default="idle")
    sync_started_at = Column(DateTime, nullable=True)
    sync_completed_at = Column(DateTime, nullable=True)
    earliest_synced_date = Column(Date, nullable=True)
    latest_synced_date = Column(Date, nullable=True)
    last_incremental_sync_at = Column(DateTime, nullable=True)
    weekly_sync_last_attempt_at = Column(DateTime, nullable=True)
    last_error = Column(String(500), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', name='uq_stock_history_sync_state_symbol'),
        Index('ix_stock_history_sync_state_symbol', 'symbol', unique=True),
        Index('ix_stock_history_sync_state_sync_status', 'sync_status'),
        Index('ix_stock_history_sync_state_latest_synced_date', 'latest_synced_date'),
    )


class StockFinancialDataCache(Base):
    """Cached company financial data snapshots from vnstock."""
    __tablename__ = "stock_financial_data_cache"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    data_type = Column(String(30), nullable=False)  # income | cashflow | balance_sheet | ratios
    period = Column(String(10), nullable=False)  # quarter | year
    lang = Column(String(10), nullable=False, default="en")
    data = Column(JSON, nullable=False, default=list)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            'symbol',
            'data_type',
            'period',
            'lang',
            name='uq_stock_financial_data_cache_key',
        ),
        Index(
            'ix_stock_financial_data_cache_lookup',
            'symbol',
            'data_type',
            'period',
            'lang',
        ),
        Index('ix_stock_financial_data_cache_symbol_updated_at', 'symbol', 'updated_at'),
    )


class StockCompanyDataCache(Base):
    """Cached company profile data snapshots from vnstock."""
    __tablename__ = "stock_company_data_cache"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    data_type = Column(String(30), nullable=False)  # overview | shareholders | officers | subsidiaries
    data = Column(JSON, nullable=False, default=list)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            'symbol',
            'data_type',
            name='uq_stock_company_data_cache_key',
        ),
        Index(
            'ix_stock_company_data_cache_lookup',
            'symbol',
            'data_type',
        ),
        Index('ix_stock_company_data_cache_symbol_updated_at', 'symbol', 'updated_at'),
    )


class FundNav(Base):
    """Historical NAV data for mutual funds."""
    __tablename__ = "fund_navs"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), nullable=False)  # Fund symbol/short_name
    date = Column(Date, nullable=False)
    nav = Column(Float, nullable=False)  # NAV per unit
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_fund_symbol_date'),
        Index('ix_fund_navs_symbol_date', 'symbol', 'date'),
    )


class FundDetailCache(Base):
    """Cached fund details (top holdings, industry holdings, asset holdings)."""
    __tablename__ = "fund_detail_cache"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), nullable=False)
    detail_type = Column(String(30), nullable=False)  # top_holding | industry_holding | asset_holding
    data = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'detail_type', name='uq_fund_detail_symbol_type'),
        Index('ix_fund_detail_symbol_type', 'symbol', 'detail_type'),
    )


class FundListing(Base):
    """Cached fund listing (open-end funds)."""
    __tablename__ = "fund_listings"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    fund_type = Column(String(50), nullable=True)
    fund_owner = Column(String(255), nullable=True)
    data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_fund_listings_symbol_type', 'symbol', 'fund_type'),
    )


class User(Base):
    """Application user account."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    download_folder = Column(String(512), nullable=True)
    company_export_category = Column(String(120), nullable=True)
    finance_export_category = Column(String(120), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class BookmarkGroup(Base):
    """User-defined group for favorite stocks."""
    __tablename__ = "bookmark_groups"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_bookmark_group_user_name"),
    )


class BookmarkStock(Base):
    """Stock membership inside a bookmark group."""
    __tablename__ = "bookmark_stocks"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("bookmark_groups.id", ondelete="CASCADE"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("group_id", "ticker", name="uq_bookmark_group_ticker"),
        Index("ix_bookmark_group_ticker", "group_id", "ticker"),
    )


class PortfolioPosition(Base):
    """User portfolio position."""
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=True)
    purchase_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_portfolio_position_user_ticker"),
    )


class TradingPosition(Base):
    """User trading position for active account-specific trades."""
    __tablename__ = "trading_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    account_label = Column(String(120), nullable=False)
    ticker = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    average_entry_cost = Column(Float, nullable=False)
    opened_date = Column(Date, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    notes = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "account_label",
            "ticker",
            name="uq_trading_position_user_account_ticker",
        ),
        Index("ix_trading_position_user_account", "user_id", "account_label"),
    )


class ScheduledSyncJob(Base):
    """Persistent scheduler configuration for recurring admin sync jobs."""
    __tablename__ = "scheduled_sync_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    sync_type = Column(String(20), nullable=False)
    sync_action = Column(String(20), nullable=False)
    index_symbol = Column(String(20), nullable=True)
    symbols = Column(JSON, nullable=False, default=list)
    date_from = Column(Date, nullable=True)
    date_to = Column(Date, nullable=True)
    auto_repair = Column(Boolean, nullable=False, default=False)
    starts_at = Column(DateTime, nullable=False)
    interval_value = Column(Integer, nullable=False)
    interval_unit = Column(String(20), nullable=False)
    timezone = Column(String(64), nullable=False, default="Asia/Ho_Chi_Minh")
    max_retries = Column(Integer, nullable=False, default=3)
    partial_success_failure_threshold_percent = Column(Integer, nullable=False, default=10)
    next_run_at = Column(DateTime, nullable=False)
    last_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_scheduled_sync_jobs_enabled_next_run", "enabled", "next_run_at"),
        Index("ix_scheduled_sync_jobs_sync_type_action", "sync_type", "sync_action"),
    )


class ScheduledSyncJobRun(Base):
    """Execution log row for each scheduled sync attempt."""
    __tablename__ = "scheduled_sync_job_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scheduled_sync_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False, default=1)
    status = Column(String(20), nullable=False)
    scheduled_for = Column(DateTime, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error = Column(String(1000), nullable=True)
    summary = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_scheduled_sync_job_runs_status_scheduled_for", "status", "scheduled_for"),
        Index("ix_scheduled_sync_job_runs_job_scheduled_for", "job_id", "scheduled_for"),
    )


class NewsSite(Base):
    """Normalized site/domain metadata for RSS and crawl news sources."""
    __tablename__ = "news_sites"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(255), nullable=False)
    homepage_url = Column(String(1000), nullable=False)
    display_name = Column(String(255), nullable=True)
    is_public = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("domain", "homepage_url", name="uq_news_site_domain_homepage"),
        Index("ix_news_sites_domain", "domain"),
        Index("ix_news_sites_public", "is_public"),
    )


class NewsSiteFeed(Base):
    """Discovered or manually entered RSS/Atom feed for a site."""
    __tablename__ = "news_site_feeds"

    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("news_sites.id", ondelete="CASCADE"), nullable=False, index=True)
    feed_url = Column(String(1000), nullable=False)
    title = Column(String(255), nullable=True)
    kind = Column(String(20), nullable=False, default="rss")
    discovery_method = Column(String(20), nullable=False, default="manual")
    validation_status = Column(String(20), nullable=False, default="pending")
    validation_error = Column(String(1000), nullable=True)
    is_public = Column(Boolean, nullable=False, default=False)
    poll_interval_minutes = Column(Integer, nullable=False, default=30)
    last_polled_at = Column(DateTime, nullable=True)
    next_poll_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    last_failure_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("feed_url", name="uq_news_site_feeds_feed_url"),
        Index("ix_news_site_feeds_status_next_poll", "validation_status", "next_poll_at"),
    )


class NewsCrawlSource(Base):
    """Static HTML crawl source configuration for a site/page."""
    __tablename__ = "news_crawl_sources"

    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("news_sites.id", ondelete="CASCADE"), nullable=False, index=True)
    listing_url = Column(String(1000), nullable=False)
    article_link_selector = Column(String(255), nullable=False)
    content_selector = Column(String(255), nullable=False)
    excerpt_selector = Column(String(255), nullable=True)
    pagination_config = Column(JSON, nullable=True)
    validation_status = Column(String(20), nullable=False, default="pending")
    validation_error = Column(String(1000), nullable=True)
    is_public = Column(Boolean, nullable=False, default=False)
    poll_interval_minutes = Column(Integer, nullable=False, default=30)
    last_polled_at = Column(DateTime, nullable=True)
    next_poll_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    last_failure_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("listing_url", "article_link_selector", name="uq_news_crawl_source_listing_selector"),
        Index("ix_news_crawl_sources_status_next_poll", "validation_status", "next_poll_at"),
    )


class NewsSourceSubscription(Base):
    """User subscription to a private RSS feed or crawl source."""
    __tablename__ = "news_source_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    site_feed_id = Column(Integer, ForeignKey("news_site_feeds.id", ondelete="CASCADE"), nullable=True, index=True)
    crawl_source_id = Column(Integer, ForeignKey("news_crawl_sources.id", ondelete="CASCADE"), nullable=True, index=True)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "site_feed_id", name="uq_news_subscription_user_feed"),
        UniqueConstraint("user_id", "crawl_source_id", name="uq_news_subscription_user_crawl"),
        Index("ix_news_source_subscriptions_user_enabled", "user_id", "enabled"),
    )


class NewsArticle(Base):
    """Canonical article content deduped across feeds and sources."""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    canonical_url = Column(String(1000), nullable=False)
    title = Column(String(500), nullable=False)
    excerpt = Column(Text, nullable=True)
    llm_summary = Column(Text, nullable=True)
    content_text = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    language = Column(String(20), nullable=True)
    content_hash = Column(String(64), nullable=True)
    image_url = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_news_articles_canonical_url"),
        Index("ix_news_articles_published_at", "published_at"),
        Index("ix_news_articles_content_hash", "content_hash"),
    )


class NewsArticleSource(Base):
    """Mapping from canonical article to its origin feeds or crawl sources."""
    __tablename__ = "news_article_sources"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False, index=True)
    site_feed_id = Column(Integer, ForeignKey("news_site_feeds.id", ondelete="CASCADE"), nullable=True, index=True)
    crawl_source_id = Column(Integer, ForeignKey("news_crawl_sources.id", ondelete="CASCADE"), nullable=True, index=True)
    article_url = Column(String(1000), nullable=False)
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("article_id", "site_feed_id", name="uq_news_article_source_article_feed"),
        UniqueConstraint("article_id", "crawl_source_id", name="uq_news_article_source_article_crawl"),
    )


class NewsArticleSemantic(Base):
    """Stored semantic labels for a news article."""
    __tablename__ = "news_article_semantics"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False, index=True)
    topics = Column(JSON, nullable=False, default=list)
    tickers = Column(JSON, nullable=False, default=list)
    sectors = Column(JSON, nullable=False, default=list)
    importance = Column(String(20), nullable=True)
    sentiment = Column(String(20), nullable=True)
    raw_payload = Column(JSON, nullable=False, default=dict)
    classified_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("article_id", name="uq_news_article_semantics_article_id"),
        Index("ix_news_article_semantics_importance", "importance"),
        Index("ix_news_article_semantics_sentiment", "sentiment"),
    )


class NewsQuickGlanceDigest(Base):
    """Cached LLM digests for timeframe-based news summaries."""
    __tablename__ = "news_quick_glance_digests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    viewer_key = Column(String(32), nullable=False)
    window_hours = Column(Integer, nullable=False)
    evidence_fingerprint = Column(String(64), nullable=False)
    payload = Column(JSON, nullable=False, default=dict)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "viewer_key",
            "window_hours",
            "evidence_fingerprint",
            name="uq_news_quick_glance_digest_viewer_window_fingerprint",
        ),
        Index(
            "ix_news_quick_glance_digest_viewer_window_generated",
            "viewer_key",
            "window_hours",
            "generated_at",
        ),
    )


class NewsUserPreference(Base):
    """Per-user topic-block profile for personalized feed filtering."""
    __tablename__ = "news_user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    blocked_topics_text = Column(Text, nullable=True)
    blocked_labels = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_news_user_preferences_user_id"),
    )


class NewsIngestionRun(Base):
    """Execution log for a feed or crawl source ingestion attempt."""
    __tablename__ = "news_ingestion_runs"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(20), nullable=False)
    site_feed_id = Column(Integer, ForeignKey("news_site_feeds.id", ondelete="CASCADE"), nullable=True, index=True)
    crawl_source_id = Column(Integer, ForeignKey("news_crawl_sources.id", ondelete="CASCADE"), nullable=True, index=True)
    status = Column(String(20), nullable=False)
    fetched_count = Column(Integer, nullable=False, default=0)
    stored_count = Column(Integer, nullable=False, default=0)
    filtered_count = Column(Integer, nullable=False, default=0)
    error = Column(String(1000), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_news_ingestion_runs_status_started_at", "status", "started_at"),
    )
