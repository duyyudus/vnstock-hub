import React, { useState, useEffect } from 'react';
import { stockApi } from '../../api/stockApi';
import type { IndexValueInfo } from '../../api/stockApi';

interface IndexBannerProps {
    index: IndexValueInfo;
}

/**
 * Single index banner card showing current value and change
 */
const IndexBanner: React.FC<IndexBannerProps> = ({ index }) => {
    const isPositive = index.change >= 0;
    const changeColor = isPositive ? 'text-success' : 'text-error';
    const changeIcon = isPositive ? '▲' : '▼';

    return (
        <div className="flex-1 min-w-[140px] bg-base-100 rounded-lg p-3 shadow-md border border-base-200 hover:shadow-lg transition-shadow">
            <div className="text-xs font-semibold text-base-content/60 mb-1 truncate">
                {index.name}
            </div>
            <div className="text-lg font-bold text-primary">
                {index.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className={`text-sm font-medium ${changeColor} flex items-center gap-1`}>
                <span>{changeIcon}</span>
                <span>{Math.abs(index.change_value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                <span>({isPositive ? '+' : ''}{index.change.toFixed(2)}%)</span>
            </div>
        </div>
    );
};

/**
 * Row of index banners showing major market indices
 */
export const IndexBanners: React.FC = () => {
    const [indices, setIndices] = useState<IndexValueInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchIndexValues = async () => {
            try {
                setLoading(true);
                const response = await stockApi.getIndexValues();
                setIndices(response.indices);
                setError(null);
            } catch (err) {
                console.error('Failed to fetch index values:', err);
                setError('Failed to load market indices');
            } finally {
                setLoading(false);
            }
        };

        fetchIndexValues();

        // Refresh every 60 seconds
        const interval = setInterval(fetchIndexValues, 60000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="flex gap-4 mb-6 overflow-x-auto pb-2">
                {[...Array(5)].map((_, i) => (
                    <div key={i} className="flex-1 min-w-[140px] bg-base-100 rounded-lg p-3 shadow-md animate-pulse">
                        <div className="h-4 bg-base-300 rounded w-16 mb-2"></div>
                        <div className="h-6 bg-base-300 rounded w-20 mb-1"></div>
                        <div className="h-4 bg-base-300 rounded w-24"></div>
                    </div>
                ))}
            </div>
        );
    }

    if (error) {
        return (
            <div className="mb-6 p-3 bg-error/10 text-error rounded-lg text-center">
                {error}
            </div>
        );
    }

    return (
        <div className="flex gap-4 mb-6 overflow-x-auto pb-2">
            {indices.map((index) => (
                <IndexBanner key={index.symbol} index={index} />
            ))}
        </div>
    );
};

export default IndexBanners;
