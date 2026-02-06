import React, { useState, useMemo, useEffect, useRef } from 'react';

export interface FundInfo {
    symbol: string;
    name: string;
    fund_type?: string;
    fund_owner?: string;
}

interface FundSelectorProps {
    /** List of available funds */
    funds: FundInfo[];
    /** Currently selected fund symbol */
    selectedFund: string | null;
    /** Callback when user selects a different fund */
    onFundChange: (symbol: string) => void;
    /** Loading state */
    loading?: boolean;
}

const GROUP_ORDER = ['Quỹ cổ phiếu', 'Quỹ trái phiếu', 'Quỹ cân bằng'] as const;

const GROUP_LABELS: Record<string, string> = {
    'Quỹ cổ phiếu': 'Stock Funds',
    'Quỹ trái phiếu': 'Bond Funds',
    'Quỹ cân bằng': 'Balanced Funds',
    OTHER: 'Other Funds',
};

interface FundGroupSelectorProps {
    label: string;
    funds: FundInfo[];
    selectedFund: string | null;
    onFundChange: (symbol: string) => void;
}

const FundGroupSelector: React.FC<FundGroupSelectorProps> = ({
    label,
    funds,
    selectedFund,
    onFundChange,
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [isFocused, setIsFocused] = useState(false);
    const [activeIndex, setActiveIndex] = useState(-1);
    const listRef = useRef<HTMLDivElement>(null);

    const filteredFunds = useMemo(() => {
        const lowerSearch = searchTerm.toLowerCase();
        return funds
            .filter((fund) =>
                fund.symbol.toLowerCase().includes(lowerSearch) ||
                fund.name.toLowerCase().includes(lowerSearch)
            )
            .sort((a, b) => a.symbol.localeCompare(b.symbol));
    }, [funds, searchTerm]);

    const selectedFundInfo = useMemo(() => {
        return funds.find(f => f.symbol === selectedFund);
    }, [funds, selectedFund]);

    useEffect(() => {
        if (activeIndex >= 0 && listRef.current) {
            const activeItem = listRef.current.querySelector(`[data-index="${activeIndex}"]`);
            if (activeItem) {
                activeItem.scrollIntoView({
                    block: 'nearest',
                    behavior: 'smooth'
                });
            }
        }
    }, [activeIndex]);

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(event.target.value);
        setIsFocused(true);
        setActiveIndex(-1);
    };

    const handleSelect = (symbol: string) => {
        onFundChange(symbol);
        setSearchTerm('');
        setIsFocused(false);
        setActiveIndex(-1);
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (!isFocused || filteredFunds.length === 0) return;

        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                setActiveIndex(prev => (prev < filteredFunds.length - 1 ? prev + 1 : 0));
                break;
            case 'ArrowUp':
                event.preventDefault();
                setActiveIndex(prev => (prev > 0 ? prev - 1 : filteredFunds.length - 1));
                break;
            case 'Enter':
                event.preventDefault();
                if (activeIndex >= 0) {
                    handleSelect(filteredFunds[activeIndex].symbol);
                } else if (filteredFunds.length > 0) {
                    handleSelect(filteredFunds[0].symbol);
                }
                break;
            case 'Escape':
                setIsFocused(false);
                setActiveIndex(-1);
                break;
        }
    };

    return (
        <div className="flex flex-col gap-1 flex-1 min-w-0">
            <span className="text-xs font-semibold text-base-content/60 uppercase tracking-wide px-1">
                {label} ({funds.length})
            </span>
            <div className="relative w-full">
                <div className="relative">
                    <input
                        type="text"
                        placeholder={selectedFundInfo ? `Current: ${selectedFundInfo.symbol}` : `Search ${label.toLowerCase()}...`}
                        className="input input-bordered input-sm w-full pr-8 focus:input-primary"
                        value={searchTerm}
                        onChange={handleSearchChange}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setTimeout(() => setIsFocused(false), 200)}
                    />
                    <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-base-content/50">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
                        </svg>
                    </div>
                </div>

                {(isFocused && (searchTerm || isFocused)) && (
                    <div
                        ref={listRef}
                        className="absolute z-[100] w-full mt-1 bg-base-100 border border-base-300 rounded-box shadow-xl max-h-60 overflow-y-auto"
                    >
                        <ul className="menu menu-compact p-2">
                            {filteredFunds.length === 0 ? (
                                <li className="disabled text-base-content/50 p-2">No matching funds</li>
                            ) : (
                                filteredFunds.map((fund, index) => (
                                    <li key={fund.symbol} data-index={index}>
                                        <button
                                            type="button"
                                            className={`flex justify-between items-center ${selectedFund === fund.symbol ? 'active' : ''} ${activeIndex === index ? 'bg-base-200' : ''}`}
                                            onClick={() => handleSelect(fund.symbol)}
                                            onMouseEnter={() => setActiveIndex(index)}
                                        >
                                            <div className="flex flex-col items-start overflow-hidden">
                                                <span className="font-bold text-sm">{fund.symbol}</span>
                                                <span className="text-xs opacity-70 truncate w-full text-left">{fund.name}</span>
                                            </div>
                                            {selectedFund === fund.symbol && (
                                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4 ml-2 flex-shrink-0">
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                                                </svg>
                                            )}
                                        </button>
                                    </li>
                                ))
                            )}
                        </ul>
                    </div>
                )}
            </div>

            {!isFocused && selectedFundInfo && (
                <div className="flex items-center gap-2 text-xs text-base-content/70 px-1">
                    <span className="badge badge-sm badge-outline badge-primary">{selectedFundInfo.symbol}</span>
                    <span className="truncate">{selectedFundInfo.name}</span>
                </div>
            )}
        </div>
    );
};

/**
 * Dropdown selector for choosing which fund to display,
 * grouped by fund type (Stock, Bond, Balanced).
 */
export const FundSelector: React.FC<FundSelectorProps> = ({
    funds,
    selectedFund,
    onFundChange,
    loading = false,
}) => {
    const groupedFunds = useMemo(() => {
        const groups: Record<string, FundInfo[]> = {};
        for (const fund of funds) {
            const key = (fund.fund_type && GROUP_ORDER.includes(fund.fund_type as typeof GROUP_ORDER[number]))
                ? fund.fund_type
                : 'OTHER';
            if (!groups[key]) groups[key] = [];
            groups[key].push(fund);
        }
        return groups;
    }, [funds]);

    const orderedKeys = useMemo(() => {
        const keys: string[] = [];
        for (const key of GROUP_ORDER) {
            if (groupedFunds[key]?.length) keys.push(key);
        }
        if (groupedFunds['OTHER']?.length) keys.push('OTHER');
        return keys;
    }, [groupedFunds]);

    if (loading) {
        return (
            <div className="flex items-center gap-2">
                <span className="loading loading-spinner loading-sm"></span>
                <span className="text-sm">Loading funds...</span>
            </div>
        );
    }

    return (
        <div className="flex flex-row gap-4 w-full">
            {orderedKeys.map((key) => (
                <FundGroupSelector
                    key={key}
                    label={GROUP_LABELS[key] || key}
                    funds={groupedFunds[key]}
                    selectedFund={selectedFund}
                    onFundChange={onFundChange}
                />
            ))}
        </div>
    );
};

export default FundSelector;
