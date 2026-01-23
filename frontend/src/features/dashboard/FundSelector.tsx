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

/**
 * Dropdown selector for choosing which fund to display.
 */
export const FundSelector: React.FC<FundSelectorProps> = ({
    funds,
    selectedFund,
    onFundChange,
    loading = false,
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

    // Reset active index when filtered list changes
    useEffect(() => {
        setActiveIndex(-1);
    }, [filteredFunds]);

    // Scroll active item into view
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
                    // If no index is focused but we hit enter, select the first one
                    handleSelect(filteredFunds[0].symbol);
                }
                break;
            case 'Escape':
                setIsFocused(false);
                setActiveIndex(-1);
                break;
        }
    };

    if (loading) {
        return (
            <div className="flex items-center gap-2">
                <span className="loading loading-spinner loading-sm"></span>
                <span className="text-sm">Loading funds...</span>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-3">
            <div className="relative w-full max-w-md">
                {/* Search Input */}
                <div className="relative">
                    <input
                        type="text"
                        placeholder={selectedFund ? `Current: ${selectedFund}` : "Search fund name or symbol..."}
                        className="input input-bordered w-full pr-10 focus:input-primary"
                        value={searchTerm}
                        onChange={handleSearchChange}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setTimeout(() => setIsFocused(false), 200)}
                    />
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-base-content/50">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
                        </svg>
                    </div>
                </div>

                {/* Dropdown Results */}
                {(isFocused && (searchTerm || isFocused)) && (
                    <div
                        ref={listRef}
                        className="absolute z-[100] w-full mt-1 bg-base-100 border border-base-300 rounded-box shadow-xl max-h-80 overflow-y-auto"
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

            {/* Selection indicator if not focused */}
            {!isFocused && selectedFundInfo && (
                <div className="flex items-center gap-2 text-xs text-base-content/70 px-1">
                    <span className="badge badge-sm badge-outline badge-primary">{selectedFundInfo.symbol}</span>
                    <span className="truncate">{selectedFundInfo.name}</span>
                </div>
            )}
        </div>
    );
};

export default FundSelector;
