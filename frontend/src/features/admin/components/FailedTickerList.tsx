import React from 'react';

interface FailedTickerListProps {
    tickers: string[];
}

export const FailedTickerList: React.FC<FailedTickerListProps> = ({ tickers }) => {
    if (tickers.length === 0) {
        return null;
    }

    return (
        <div className="space-y-2">
            <p className="text-sm font-medium">Failed tickers</p>
            <div className="max-h-24 overflow-y-auto rounded-box bg-base-200/60 p-2">
                <div className="flex flex-wrap gap-2">
                    {tickers.map((ticker) => (
                        <span key={ticker} className="badge badge-error badge-outline">
                            {ticker}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
};
