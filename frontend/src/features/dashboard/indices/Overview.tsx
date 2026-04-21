import React, { useMemo } from 'react';
import type { Stock, BookmarkGroup } from '../../../api/stockApi';
import { StocksTable } from './StocksTable';
import type {
    PortfolioHoldingSummary,
    StockIndustrySection,
    TradingPositionSummary,
} from './StocksTable';

interface OverviewProps {
    /** List of stocks to display in the overview */
    stocks: Stock[];
    /** Aggregate foreign net value for the selected index universe */
    foreignNetSummaryValue?: number | null;
    /** Label for the selected index summary */
    foreignNetSummaryLabel?: string;
    /** Bookmark groups for the logged-in user */
    bookmarkGroups?: BookmarkGroup[];
    /** Current holding details keyed by uppercase ticker */
    portfolioHoldings?: Record<string, PortfolioHoldingSummary>;
    /** Open trading position summaries keyed by uppercase ticker */
    openTradingPositions?: Record<string, TradingPositionSummary>;
    /** Notify parent to refresh bookmark data */
    onBookmarksUpdated?: (groupId?: number) => void;
}

const normalizeIndustryLabel = (industry: string | null | undefined): string => {
    const normalized = industry?.trim() || '';
    return normalized || 'Other';
};

const buildIndustrySections = (stocks: Stock[]): StockIndustrySection[] => {
    const groupedStocks = new Map<string, { totalMarketCap: number; stocks: Stock[] }>();

    stocks.forEach((stock) => {
        const industry = normalizeIndustryLabel(stock.industry);
        const currentSection = groupedStocks.get(industry) ?? {
            totalMarketCap: 0,
            stocks: [],
        };
        const marketCap = Number(stock.market_cap);
        currentSection.stocks.push(stock);
        if (Number.isFinite(marketCap)) {
            currentSection.totalMarketCap += marketCap;
        }
        groupedStocks.set(industry, currentSection);
    });

    return Array.from(groupedStocks.entries())
        .map(([industry, section]) => ({
            industry,
            totalMarketCap: section.totalMarketCap,
            stocks: section.stocks,
        }))
        .sort((a, b) => {
            if (b.totalMarketCap !== a.totalMarketCap) {
                return b.totalMarketCap - a.totalMarketCap;
            }
            return a.industry.localeCompare(b.industry);
        });
};

export const Overview: React.FC<OverviewProps> = ({
    stocks,
    foreignNetSummaryValue = undefined,
    foreignNetSummaryLabel,
    bookmarkGroups = [],
    portfolioHoldings = {},
    openTradingPositions = {},
    onBookmarksUpdated,
}) => {
    const industrySections = useMemo(() => buildIndustrySections(stocks), [stocks]);

    return (
        <div className="space-y-3">
            <StocksTable
                stocks={stocks}
                industrySections={industrySections}
                foreignNetSummaryValue={foreignNetSummaryValue}
                foreignNetSummaryLabel={foreignNetSummaryLabel}
                bookmarkGroups={bookmarkGroups}
                portfolioHoldings={portfolioHoldings}
                openTradingPositions={openTradingPositions}
                onBookmarksUpdated={onBookmarksUpdated}
            />
        </div>
    );
};

export default Overview;
