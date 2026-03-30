import React from 'react';
import type { SyncCollectionScope } from '../adminUtils';

interface SyncCollectionSelectorProps {
    value: SyncCollectionScope;
    onChange: (value: SyncCollectionScope) => void;
    portfolioCount: number;
    tradingCount: number;
}

export const SyncCollectionSelector: React.FC<SyncCollectionSelectorProps> = ({
    value,
    onChange,
    portfolioCount,
    tradingCount,
}) => {
    return (
        <label className="form-control">
            <span className="label-text">Collection scope (optional)</span>
            <select
                className="select select-bordered"
                value={value}
                onChange={(event) => onChange(event.target.value as SyncCollectionScope)}
            >
                <option value="manual">Manual symbols / index scope</option>
                <option value="portfolio" disabled={portfolioCount === 0}>
                    Portfolio holdings ({portfolioCount})
                </option>
                <option value="trading" disabled={tradingCount === 0}>
                    Trading positions ({tradingCount})
                </option>
            </select>
        </label>
    );
};

export default SyncCollectionSelector;
