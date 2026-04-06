import React from 'react';

type MarkdownContentProps = {
    content: string;
    className?: string;
};

type MarkdownBlock =
    | { type: 'heading'; level: number; content: string }
    | { type: 'paragraph'; content: string }
    | { type: 'unordered-list'; items: string[] }
    | { type: 'ordered-list'; items: string[] }
    | { type: 'blockquote'; content: string[] }
    | { type: 'code'; language: string; content: string };

const INLINE_TOKEN_PATTERN = /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|_([^_]+)_)/;
const UNORDERED_LIST_PATTERN = /^[-*+•]\s+/;
const UNORDERED_LIST_ITEM_PATTERN = /^[-*+•]\s+(.*)$/;

const joinClassNames = (...values: Array<string | undefined>) => values.filter(Boolean).join(' ');

const renderInline = (content: string, keyPrefix: string): React.ReactNode[] => {
    const nodes: React.ReactNode[] = [];
    let remaining = content;
    let tokenIndex = 0;

    while (remaining.length > 0) {
        const match = remaining.match(INLINE_TOKEN_PATTERN);
        if (!match || match.index === undefined) {
            nodes.push(remaining);
            break;
        }

        if (match.index > 0) {
            nodes.push(remaining.slice(0, match.index));
        }

        if (match[2] && match[3]) {
            nodes.push(
                <a
                    key={`${keyPrefix}-link-${tokenIndex}`}
                    href={match[3]}
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary underline underline-offset-2 hover:no-underline"
                >
                    {renderInline(match[2], `${keyPrefix}-link-text-${tokenIndex}`)}
                </a>,
            );
        } else if (match[4]) {
            nodes.push(
                <code key={`${keyPrefix}-code-${tokenIndex}`} className="rounded bg-base-300 px-1.5 py-0.5 font-mono text-[0.92em]">
                    {match[4]}
                </code>,
            );
        } else if (match[5] || match[6]) {
            const strongContent = match[5] || match[6] || '';
            nodes.push(
                <strong key={`${keyPrefix}-strong-${tokenIndex}`} className="font-semibold text-base-content">
                    {renderInline(strongContent, `${keyPrefix}-strong-text-${tokenIndex}`)}
                </strong>,
            );
        } else if (match[7] || match[8]) {
            const emphasisContent = match[7] || match[8] || '';
            nodes.push(
                <em key={`${keyPrefix}-em-${tokenIndex}`} className="italic">
                    {renderInline(emphasisContent, `${keyPrefix}-em-text-${tokenIndex}`)}
                </em>,
            );
        }

        remaining = remaining.slice(match.index + match[0].length);
        tokenIndex += 1;
    }

    return nodes;
};

const parseMarkdownBlocks = (content: string): MarkdownBlock[] => {
    const normalized = content.replace(/\r\n/g, '\n').trim();
    if (!normalized) {
        return [];
    }

    const lines = normalized.split('\n');
    const blocks: MarkdownBlock[] = [];
    let index = 0;

    while (index < lines.length) {
        const line = lines[index];
        const trimmed = line.trim();

        if (!trimmed) {
            index += 1;
            continue;
        }

        if (trimmed.startsWith('```')) {
            const language = trimmed.slice(3).trim();
            const codeLines: string[] = [];
            index += 1;

            while (index < lines.length && !lines[index].trim().startsWith('```')) {
                codeLines.push(lines[index]);
                index += 1;
            }

            if (index < lines.length) {
                index += 1;
            }

            blocks.push({
                type: 'code',
                language,
                content: codeLines.join('\n'),
            });
            continue;
        }

        const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
            blocks.push({
                type: 'heading',
                level: headingMatch[1].length,
                content: headingMatch[2].trim(),
            });
            index += 1;
            continue;
        }

        if (trimmed.startsWith('>')) {
            const quoteLines: string[] = [];
            while (index < lines.length) {
                const quoteLine = lines[index].trim();
                if (!quoteLine.startsWith('>')) {
                    break;
                }
                quoteLines.push(quoteLine.replace(/^>\s?/, ''));
                index += 1;
            }
            blocks.push({ type: 'blockquote', content: quoteLines });
            continue;
        }

        if (UNORDERED_LIST_PATTERN.test(trimmed)) {
            const items: string[] = [];
            while (index < lines.length) {
                const listLine = lines[index].trim();
                const listMatch = listLine.match(UNORDERED_LIST_ITEM_PATTERN);
                if (!listMatch) {
                    break;
                }
                items.push(listMatch[1]);
                index += 1;
            }
            blocks.push({ type: 'unordered-list', items });
            continue;
        }

        if (/^\d+\.\s+/.test(trimmed)) {
            const items: string[] = [];
            while (index < lines.length) {
                const listLine = lines[index].trim();
                const listMatch = listLine.match(/^\d+\.\s+(.*)$/);
                if (!listMatch) {
                    break;
                }
                items.push(listMatch[1]);
                index += 1;
            }
            blocks.push({ type: 'ordered-list', items });
            continue;
        }

        const paragraphLines: string[] = [];
        while (index < lines.length) {
            const paragraphLine = lines[index];
            const paragraphTrimmed = paragraphLine.trim();
            if (
                !paragraphTrimmed ||
                paragraphTrimmed.startsWith('```') ||
                /^#{1,6}\s+/.test(paragraphTrimmed) ||
                paragraphTrimmed.startsWith('>') ||
                UNORDERED_LIST_PATTERN.test(paragraphTrimmed) ||
                /^\d+\.\s+/.test(paragraphTrimmed)
            ) {
                break;
            }
            paragraphLines.push(paragraphTrimmed);
            index += 1;
        }

        if (paragraphLines.length > 0) {
            blocks.push({
                type: 'paragraph',
                content: paragraphLines.join(' '),
            });
            continue;
        }

        index += 1;
    }

    return blocks;
};

const renderBlock = (block: MarkdownBlock, index: number) => {
    switch (block.type) {
        case 'heading': {
            const headingClassName =
                block.level <= 2
                    ? 'text-base font-semibold text-base-content'
                    : 'text-sm font-semibold uppercase tracking-[0.08em] text-base-content/80';
            return (
                <h4 key={`heading-${index}`} className={headingClassName}>
                    {renderInline(block.content, `heading-${index}`)}
                </h4>
            );
        }
        case 'paragraph':
            return (
                <p key={`paragraph-${index}`} className="leading-6 text-base-content/85">
                    {renderInline(block.content, `paragraph-${index}`)}
                </p>
            );
        case 'unordered-list':
            return (
                <ul key={`unordered-${index}`} className="list-disc space-y-1 pl-5 text-base-content/85">
                    {block.items.map((item, itemIndex) => (
                        <li key={`unordered-${index}-${itemIndex}`}>{renderInline(item, `unordered-${index}-${itemIndex}`)}</li>
                    ))}
                </ul>
            );
        case 'ordered-list':
            return (
                <ol key={`ordered-${index}`} className="list-decimal space-y-1 pl-5 text-base-content/85">
                    {block.items.map((item, itemIndex) => (
                        <li key={`ordered-${index}-${itemIndex}`}>{renderInline(item, `ordered-${index}-${itemIndex}`)}</li>
                    ))}
                </ol>
            );
        case 'blockquote':
            return (
                <blockquote key={`quote-${index}`} className="border-l-4 border-base-300 pl-4 italic text-base-content/70">
                    {block.content.map((line, lineIndex) => (
                        <p key={`quote-${index}-${lineIndex}`}>{renderInline(line, `quote-${index}-${lineIndex}`)}</p>
                    ))}
                </blockquote>
            );
        case 'code':
            return (
                <div key={`code-${index}`} className="overflow-x-auto rounded-xl border border-base-300 bg-base-200/80">
                    {block.language ? (
                        <div className="border-b border-base-300 px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-base-content/55">
                            {block.language}
                        </div>
                    ) : null}
                    <pre className="p-3 text-xs leading-6 text-base-content">
                        <code>{block.content}</code>
                    </pre>
                </div>
            );
        default:
            return null;
    }
};

const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, className }) => {
    const blocks = parseMarkdownBlocks(content);

    if (blocks.length === 0) {
        return null;
    }

    return (
        <div className={joinClassNames('space-y-3 text-sm', className)}>
            {blocks.map((block, index) => renderBlock(block, index))}
        </div>
    );
};

export default MarkdownContent;
