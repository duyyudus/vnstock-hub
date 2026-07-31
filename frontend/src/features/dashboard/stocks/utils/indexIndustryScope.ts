import type { Stock, IndustryInfo } from '../../../../api/stockApi';

interface IndustryAllocation {
    industry: string;
    allocation: number;
}

interface IndexIndustryScope {
    selectorIndustries: IndustryInfo[];
    industryAllocation: IndustryAllocation[];
    allowedIndustryNames: Set<string>;
}

const normalizeIndustryName = (industryName: string | null | undefined): string => {
    return industryName?.trim() || '';
};

const buildFallbackIndustryInfo = (industryName: string): IndustryInfo => {
    return {
        name: industryName,
        en_name: industryName,
        code: `derived-${encodeURIComponent(industryName)}`,
    };
};

export const deriveIndexIndustryScope = (
    indexUniverseStocks: Stock[],
    allIndustries: IndustryInfo[]
): IndexIndustryScope => {
    const allowedIndustryNames = new Set<string>();
    const marketCapByIndustry = new Map<string, number>();
    let totalMarketCap = 0;

    indexUniverseStocks.forEach((stock) => {
        const normalizedIndustry = normalizeIndustryName(stock.industry);
        if (normalizedIndustry) {
            allowedIndustryNames.add(normalizedIndustry);
        }

        const marketCap = Number(stock.market_cap);
        if (!Number.isFinite(marketCap) || marketCap <= 0) {
            return;
        }

        const chartIndustry = normalizedIndustry || 'Other';
        marketCapByIndustry.set(chartIndustry, (marketCapByIndustry.get(chartIndustry) || 0) + marketCap);
        totalMarketCap += marketCap;
    });

    const selectorIndustries = allIndustries.filter((industry) => {
        return allowedIndustryNames.has(normalizeIndustryName(industry.name));
    });

    const existingIndustryNames = new Set(selectorIndustries.map((industry) => normalizeIndustryName(industry.name)));
    Array.from(allowedIndustryNames)
        .sort((a, b) => a.localeCompare(b))
        .forEach((industryName) => {
            if (!existingIndustryNames.has(industryName)) {
                selectorIndustries.push(buildFallbackIndustryInfo(industryName));
            }
        });

    const industryAllocation = totalMarketCap > 0
        ? Array.from(marketCapByIndustry.entries())
            .map(([industry, marketCap]) => ({
                industry,
                allocation: (marketCap / totalMarketCap) * 100,
            }))
            .sort((a, b) => b.allocation - a.allocation)
        : [];

    return {
        selectorIndustries,
        industryAllocation,
        allowedIndustryNames,
    };
};
