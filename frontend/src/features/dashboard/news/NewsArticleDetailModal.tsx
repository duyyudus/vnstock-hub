import React, { useEffect, useRef } from 'react';
import { formatDateTime } from '../../admin/adminUtils';
import {
    type NewsArticleDetail,
    type NewsDiscussionCitation,
    type NewsDiscussionMessage,
    type NewsDiscussionSearchMode,
} from '../../../api/stockApi';
import MarkdownContent from './MarkdownContent';

export type DetailView = 'article' | 'discussion';

export type DiscussionThreadMessage = {
    id: string;
    role: NewsDiscussionMessage['role'];
    content: string;
    citations: NewsDiscussionCitation[];
    warning: string | null;
    searchMode: NewsDiscussionSearchMode;
    effectiveSearchMode: 'off' | 'on';
    usedWebSearch: boolean;
    webResultsCount: number;
};

type NewsArticleDetailModalProps = {
    article: NewsArticleDetail | null;
    isSignedIn: boolean;
    detailView: DetailView;
    detailLoadingId: number | null;
    detailError: string | null;
    detailRefreshLoading: boolean;
    discussionMessages: DiscussionThreadMessage[];
    discussionDraft: string;
    discussionSearchMode: NewsDiscussionSearchMode;
    discussionQueryOverride: string;
    discussionLoading: boolean;
    discussionError: string | null;
    onClose: () => void;
    onRefreshArticleContent: () => Promise<void> | void;
    onDetailViewChange: (view: DetailView) => void;
    onOpenRelatedArticle: (articleId: number, view?: DetailView) => Promise<void> | void;
    onDiscussionSearchModeChange: (mode: NewsDiscussionSearchMode) => void;
    onDiscussionQueryOverrideChange: (value: string) => void;
    onDiscussionDraftChange: (value: string) => void;
    onDiscussionInputKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
    onSubmitDiscussion: () => Promise<void> | void;
};

const joinTags = (values: string[]) => values.join(' · ');

const renderCitationGroup = (
    messageId: string,
    label: string,
    badgeLabel: string,
    citations: NewsDiscussionCitation[],
) => {
    if (citations.length === 0) {
        return null;
    }

    return (
        <div className="space-y-2">
            <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-base-content/50">{label}</p>
            <div className="space-y-2">
                {citations.map((citation, index) => (
                    <div key={`${messageId}-${badgeLabel}-${index}`} className="rounded-xl border border-base-300 bg-base-100 p-3 text-xs text-base-content/70">
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="badge badge-outline badge-xs">{badgeLabel}</span>
                            <span className="font-medium text-base-content">{citation.title}</span>
                            {citation.domain ? <span>{citation.domain}</span> : null}
                        </div>
                        <p className="mt-2 leading-5">{citation.snippet}</p>
                        {citation.url ? (
                            <a href={citation.url} target="_blank" rel="noreferrer" className="mt-2 inline-flex text-primary underline-offset-2 hover:underline">
                                Open source
                            </a>
                        ) : null}
                    </div>
                ))}
            </div>
        </div>
    );
};

const NewsArticleDetailModal: React.FC<NewsArticleDetailModalProps> = ({
    article,
    isSignedIn,
    detailView,
    detailLoadingId,
    detailError,
    detailRefreshLoading,
    discussionMessages,
    discussionDraft,
    discussionSearchMode,
    discussionQueryOverride,
    discussionLoading,
    discussionError,
    onClose,
    onRefreshArticleContent,
    onDetailViewChange,
    onOpenRelatedArticle,
    onDiscussionSearchModeChange,
    onDiscussionQueryOverrideChange,
    onDiscussionDraftChange,
    onDiscussionInputKeyDown,
    onSubmitDiscussion,
}) => {
    const dialogRef = useRef<HTMLDialogElement>(null);
    const showArticleView = !isSignedIn || detailView === 'article';
    const showDiscussionView = isSignedIn && detailView === 'discussion';

    useEffect(() => {
        const dialog = dialogRef.current;
        if (!dialog) {
            return;
        }
        if (article) {
            if (!dialog.open) {
                dialog.showModal();
            }
            return;
        }
        if (dialog.open) {
            dialog.close();
        }
    }, [article]);

    return (
        <dialog ref={dialogRef} className="modal" onClose={onClose}>
            <div className="modal-box max-w-6xl">
                {article ? (
                    <div className="space-y-4">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                            <div className="space-y-2">
                                <div className="flex flex-wrap items-center gap-2">
                                    <h3 className="card-title text-2xl">{article.title}</h3>
                                    {article.source_kind ? <span className="badge badge-ghost">{article.source_kind}</span> : null}
                                </div>
                                <p className="text-sm text-base-content/70">
                                    {article.source_title || joinTags(article.source_labels)}
                                    {article.published_at ? ` · ${formatDateTime(article.published_at)}` : ''}
                                </p>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <button
                                    type="button"
                                    className="btn btn-outline btn-sm"
                                    onClick={() => void onRefreshArticleContent()}
                                    disabled={detailRefreshLoading}
                                >
                                    {detailRefreshLoading ? 'Refreshing...' : 'Refresh content'}
                                </button>
                                <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>
                                    Close
                                </button>
                            </div>
                        </div>

                        {isSignedIn ? (
                            <div role="tablist" className="tabs tabs-boxed">
                                <button
                                    type="button"
                                    role="tab"
                                    className={`tab ${detailView === 'article' ? 'tab-active' : ''}`}
                                    onClick={() => onDetailViewChange('article')}
                                >
                                    Article
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    className={`tab ${detailView === 'discussion' ? 'tab-active' : ''}`}
                                    onClick={() => onDetailViewChange('discussion')}
                                >
                                    Discuss
                                </button>
                            </div>
                        ) : null}

                        {detailError ? (
                            <div className="alert alert-error">
                                <span>{detailError}</span>
                            </div>
                        ) : null}

                        {showArticleView ? (
                            <div className="space-y-4">
                                {article.content_text ? (
                                    <div className="max-h-[55vh] overflow-y-auto rounded-2xl border border-base-300 bg-base-100/80 p-4">
                                        <p className="whitespace-pre-wrap leading-7 text-base-content/85">{article.content_text}</p>
                                    </div>
                                ) : (
                                    <p className="text-sm text-base-content/60">No article body was returned.</p>
                                )}

                                {article.why_relevant.length > 0 ? (
                                    <div className="flex flex-wrap gap-2">
                                        {article.why_relevant.map((reason) => (
                                            <span key={reason} className="badge badge-primary badge-outline">{reason}</span>
                                        ))}
                                    </div>
                                ) : null}

                                <div className="flex flex-wrap gap-2">
                                    {article.event_labels.map((label) => (
                                        <span key={label} className="badge badge-warning badge-outline">{label}</span>
                                    ))}
                                    {article.topics.map((topic) => (
                                        <span key={topic} className="badge badge-outline">{topic}</span>
                                    ))}
                                    {article.matched_tickers.map((ticker) => (
                                        <span key={`matched-${ticker}`} className="badge badge-primary">{ticker}</span>
                                    ))}
                                    {article.tickers.map((ticker) => (
                                        <span key={ticker} className="badge badge-secondary badge-outline">{ticker}</span>
                                    ))}
                                    {article.sectors.map((sector) => (
                                        <span key={sector} className="badge badge-accent badge-outline">{sector}</span>
                                    ))}
                                </div>

                                <div className="flex flex-wrap gap-3 text-xs text-base-content/60">
                                    {article.importance ? <span>Importance: {article.importance}</span> : null}
                                    {article.sentiment ? <span>Sentiment: {article.sentiment}</span> : null}
                                    {article.story_source_count > 1 ? <span>Story coverage: {article.story_source_count} sources</span> : null}
                                </div>

                                <div className="flex flex-wrap items-center gap-3">
                                    <a href={article.canonical_url} target="_blank" rel="noreferrer" className="btn btn-primary btn-sm">
                                        Open source article
                                    </a>
                                    <div className="text-xs text-base-content/60">
                                        Source URLs: {article.source_urls.join(' · ')}
                                    </div>
                                </div>

                                {article.related_articles.length > 0 ? (
                                    <div className="space-y-3 rounded-2xl border border-base-300 bg-base-100/80 p-4">
                                        <div className="flex flex-wrap items-center justify-between gap-2">
                                            <h4 className="font-semibold text-base-content">Related Coverage</h4>
                                            <span className="badge badge-outline">{article.related_articles.length}</span>
                                        </div>
                                        <div className="space-y-2">
                                            {article.related_articles.map((related) => (
                                                <div key={related.id} className="flex flex-col gap-2 rounded-xl border border-base-300 p-3 md:flex-row md:items-center md:justify-between">
                                                    <div className="min-w-0">
                                                        <p className="font-medium text-base-content">{related.title}</p>
                                                        <p className="text-xs text-base-content/60">
                                                            {related.source_title || 'Unknown source'}
                                                            {related.published_at ? ` · ${formatDateTime(related.published_at)}` : ''}
                                                        </p>
                                                    </div>
                                                    <div className="flex flex-wrap gap-2">
                                                        <button
                                                            type="button"
                                                            className="btn btn-sm btn-outline"
                                                            onClick={() => void onOpenRelatedArticle(related.id)}
                                                            disabled={detailLoadingId === related.id}
                                                        >
                                                            {detailLoadingId === related.id ? 'Opening...' : 'Open detail'}
                                                        </button>
                                                        {isSignedIn ? (
                                                            <button
                                                                type="button"
                                                                className="btn btn-sm btn-outline"
                                                                onClick={() => void onOpenRelatedArticle(related.id, 'discussion')}
                                                                disabled={detailLoadingId === related.id}
                                                            >
                                                                {detailLoadingId === related.id ? 'Opening...' : 'Discuss'}
                                                            </button>
                                                        ) : null}
                                                        <a href={related.canonical_url} target="_blank" rel="noreferrer" className="btn btn-sm btn-ghost">
                                                            Open source
                                                        </a>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : null}
                            </div>
                        ) : null}

                        {showDiscussionView ? (
                            <aside className="space-y-4 rounded-2xl border border-base-300 bg-base-100/80 p-4">
                                    <div className="space-y-1">
                                        <h4 className="font-semibold text-base-content">Discuss This Article</h4>
                                        <p className="text-sm text-base-content/65">
                                            Answers are grounded in the article by default. Enable web search only when you want fresh outside context.
                                        </p>
                                    </div>

                                    {discussionError ? (
                                        <div className="alert alert-error">
                                            <span>{discussionError}</span>
                                        </div>
                                    ) : null}

                                    <div className="max-h-[55vh] space-y-3 overflow-y-auto rounded-2xl border border-base-300 bg-base-100 p-3">
                                        {discussionMessages.length === 0 ? (
                                            <div className="rounded-xl border border-dashed border-base-300 p-4 text-sm text-base-content/65">
                                                Ask about the article, request a recap of implications, or turn on web search for broader context.
                                            </div>
                                        ) : null}

                                        {discussionMessages.map((message) => {
                                            const articleCitations = message.citations.filter((citation) => citation.source_type === 'article');
                                            const webCitations = message.citations.filter((citation) => citation.source_type === 'web');

                                            return (
                                                <div
                                                    key={message.id}
                                                    className={`space-y-3 rounded-2xl border p-3 ${
                                                        message.role === 'assistant'
                                                            ? 'border-primary/20 bg-primary/5'
                                                            : 'border-base-300 bg-base-200/70'
                                                    }`}
                                                >
                                                    <div className="flex items-center justify-between gap-2">
                                                        <span className="text-xs font-medium uppercase tracking-[0.18em] text-base-content/55">
                                                            {message.role === 'assistant' ? 'Assistant' : 'You'}
                                                        </span>
                                                        {message.role === 'assistant' ? (
                                                            <div className="flex flex-wrap gap-2">
                                                                {message.searchMode === 'auto' ? (
                                                                    <span className="badge badge-outline badge-sm">
                                                                        Auto: {message.effectiveSearchMode === 'on' ? 'searched' : 'article only'}
                                                                    </span>
                                                                ) : null}
                                                                {message.usedWebSearch ? (
                                                                    <span className="badge badge-outline badge-sm">Web + article</span>
                                                                ) : null}
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                    {message.role === 'assistant' ? (
                                                        <MarkdownContent content={message.content} />
                                                    ) : (
                                                        <p className="whitespace-pre-wrap text-sm leading-6 text-base-content/85">{message.content}</p>
                                                    )}

                                                    {message.role === 'assistant' && message.citations.length > 0 ? (
                                                        <div className="space-y-3">
                                                            {renderCitationGroup(message.id, 'Article context', 'Article', articleCitations)}
                                                            {renderCitationGroup(message.id, 'Web citations', 'Web', webCitations)}
                                                        </div>
                                                    ) : null}

                                                    {message.role === 'assistant' && message.warning ? (
                                                        <div className="rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-base-content/80">
                                                            {message.warning}
                                                        </div>
                                                    ) : null}
                                                </div>
                                            );
                                        })}

                                        {discussionLoading ? (
                                            <div className="rounded-2xl border border-base-300 bg-base-100 p-4 text-sm text-base-content/65">
                                                <span className="loading loading-dots loading-sm text-primary" /> Generating grounded response...
                                            </div>
                                        ) : null}
                                    </div>

                                    <div className="space-y-3 rounded-2xl border border-base-300 bg-base-100 p-3">
                                        <label className="form-control">
                                            <div className="label">
                                                <span className="label-text">Search mode</span>
                                            </div>
                                            <select
                                                className="select select-bordered select-sm"
                                                value={discussionSearchMode}
                                                onChange={(event) => onDiscussionSearchModeChange(event.target.value as NewsDiscussionSearchMode)}
                                                disabled={discussionLoading}
                                            >
                                                <option value="off">Off: article only</option>
                                                <option value="auto">Auto: decide from question</option>
                                                <option value="on">On: always search</option>
                                            </select>
                                            <div className="label">
                                                <span className="label-text-alt">
                                                    Auto searches for broader-context questions like overview, ownership, comparison, or latest updates.
                                                </span>
                                            </div>
                                        </label>

                                        {discussionSearchMode !== 'off' ? (
                                            <label className="form-control">
                                                <div className="label">
                                                    <span className="label-text">Optional custom search query</span>
                                                </div>
                                                <input
                                                    type="text"
                                                    className="input input-bordered input-sm"
                                                    value={discussionQueryOverride}
                                                    onChange={(event) => onDiscussionQueryOverrideChange(event.target.value)}
                                                    placeholder="Optional: override the search query"
                                                    disabled={discussionLoading}
                                                />
                                            </label>
                                        ) : null}

                                        <label className="form-control">
                                            <div className="label">
                                                <span className="label-text">Your message</span>
                                            </div>
                                            <textarea
                                                className="textarea textarea-bordered min-h-28"
                                                value={discussionDraft}
                                                onChange={(event) => onDiscussionDraftChange(event.target.value)}
                                                onKeyDown={onDiscussionInputKeyDown}
                                                placeholder="Ask about what changed, what matters, or where this fits in the broader story."
                                                disabled={discussionLoading}
                                            />
                                        </label>

                                        <div className="flex items-center justify-between gap-3">
                                            <p className="text-xs text-base-content/55">
                                                Ephemeral session only. Messages reset when this modal closes.
                                            </p>
                                            <button
                                                type="button"
                                                className="btn btn-primary btn-sm"
                                                onClick={() => void onSubmitDiscussion()}
                                                disabled={discussionLoading || !discussionDraft.trim()}
                                            >
                                                {discussionLoading ? 'Sending...' : 'Send'}
                                            </button>
                                        </div>
                                    </div>
                                </aside>
                        ) : null}
                    </div>
                ) : (
                    <div className="py-6 text-sm text-base-content/60">No article detail loaded.</div>
                )}
            </div>
            <form method="dialog" className="modal-backdrop">
                <button onClick={onClose}>close</button>
            </form>
        </dialog>
    );
};

export default NewsArticleDetailModal;
