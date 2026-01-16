import React, { useEffect, useState } from 'react';
import { stockApi } from '../../api/stockApi';
import type { Stock } from '../../api/stockApi';

/**
 * VN-100 stocks table component.
 * Displays top 100 stocks by market cap with price and market cap columns.
 */
export const VN100Table: React.FC = () => {
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);
                const response = await stockApi.getVN100Stocks();
                setStocks(response.stocks);
            } catch (err) {
                setError('Failed to fetch VN-100 stocks data. Please try again.');
                console.error('Error fetching VN-100 data:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    // Format price to VND
    const formatPrice = (price: number): string => {
        return new Intl.NumberFormat('vi-VN').format(price);
    };

    // Format market cap (in billion VND)
    const formatMarketCap = (marketCap: number): string => {
        return new Intl.NumberFormat('vi-VN', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(marketCap);
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-64 gap-4">
                <span className="loading loading-spinner loading-lg text-primary"></span>
                <p className="text-base-content/70">Loading VN-100 stocks...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="alert alert-error">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="stroke-current shrink-0 h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                </svg>
                <span>{error}</span>
                <button
                    className="btn btn-sm btn-ghost"
                    onClick={() => window.location.reload()}
                >
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-base-content">VN-100 Stocks</h2>
                    <p className="text-base-content/60 text-sm">
                        Top 100 stocks by market capitalization
                    </p>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto rounded-xl border border-base-300 bg-base-100">
                <table className="table table-zebra">
                    <thead className="bg-base-200">
                        <tr>
                            <th className="text-base-content font-bold">#</th>
                            <th className="text-base-content font-bold">Ticker</th>
                            <th className="text-base-content font-bold text-right">
                                Price (VND)
                            </th>
                            <th className="text-base-content font-bold text-right">
                                Market Cap (Billion VND)
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {stocks.map((stock, index) => (
                            <tr key={stock.ticker} className="hover">
                                <td className="text-base-content/60">{index + 1}</td>
                                <td>
                                    <span className="font-bold text-base-content uppercase">
                                        {stock.ticker.slice(0, 3)}
                                    </span>
                                </td>
                                <td className="text-right font-mono text-base-content">
                                    {formatPrice(stock.price)}
                                </td>
                                <td className="text-right font-mono text-base-content">
                                    {formatMarketCap(stock.market_cap)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default VN100Table;
