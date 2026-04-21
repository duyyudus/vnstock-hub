import React from 'react';

export type PositionsFilter = 'all' | 'portfolio' | 'trading';

interface PositionsSelectorProps {
    selectedFilter: PositionsFilter;
    onFilterChange: (filter: PositionsFilter) => void;
    hasPortfolioPositions: boolean;
    hasTradingPositions: boolean;
}

export const PositionsSelector: React.FC<PositionsSelectorProps> = ({
    selectedFilter,
    onFilterChange,
    hasPortfolioPositions,
    hasTradingPositions,
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        onFilterChange(event.target.value as PositionsFilter);
    };

    return (
        <select
            className="select select-bordered select-sm w-40 bg-base-100 font-medium"
            value={selectedFilter}
            onChange={handleChange}
            aria-label="Filter stocks by saved positions"
        >
            <option value="all">-- Positions --</option>
            <option value="portfolio" disabled={!hasPortfolioPositions}>
                Portfolio
            </option>
            <option value="trading" disabled={!hasTradingPositions}>
                Trading
            </option>
        </select>
    );
};

export default PositionsSelector;
