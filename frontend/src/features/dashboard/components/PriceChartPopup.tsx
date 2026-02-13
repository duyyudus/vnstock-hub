import React, { useState, useEffect, useRef } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { PriceHistoryResponse } from '../../../api/stockApi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PRICE_RANGE_OPTIONS = [
    { label: '30D', days: 30 },
    { label: '90D', days: 90 },
    { label: '180D', days: 180 },
    { label: '365D', days: 365 },
] as const;

interface Position {
    x: number;
    y: number;
}

interface Size {
    width: number;
    height: number;
}

interface PriceChartPopupProps {
    ticker: string;
    companyName: string;
    initialPosition: Position;
    onClose: () => void;
    zIndex: number;
    onFocus: () => void;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: Array<{ payload: { date: string; close: number } }>;
}

export const PriceChartPopup: React.FC<PriceChartPopupProps> = ({
    ticker,
    companyName,
    initialPosition,
    onClose,
    zIndex,
    onFocus,
}) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [priceData, setPriceData] = useState<PriceHistoryResponse | null>(null);
    const [selectedRangeDays, setSelectedRangeDays] = useState(90);
    const [position, setPosition] = useState<Position>(initialPosition);
    const [size, setSize] = useState<Size>({ width: 700, height: 450 });
    const isDragging = useRef(false);
    const isResizing = useRef(false);
    const dragOffset = useRef<Position>({ x: 0, y: 0 });
    const resizeStart = useRef<{ x: number, y: number, w: number, h: number }>({ x: 0, y: 0, w: 0, h: 0 });
    const popupRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchPriceData = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await stockApi.getPriceHistory(ticker, selectedRangeDays);
                setPriceData(response);
            } catch (err) {
                console.error(`Error fetching price history for ${ticker}:`, err);
                setError('Failed to load price data.');
            } finally {
                setLoading(false);
            }
        };

        fetchPriceData();
    }, [ticker, selectedRangeDays]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                const popups = Array.from(document.querySelectorAll('.price-chart-popup')) as HTMLElement[];
                if (popups.length === 0) return;

                let highestZ = -1;
                popups.forEach((p) => {
                    const z = parseInt(p.style.zIndex || '0', 10);
                    if (z > highestZ) highestZ = z;
                });

                if (zIndex === highestZ) {
                    onClose();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose, zIndex]);

    const handleMouseDown = (e: React.MouseEvent) => {
        onFocus();
        const target = e.target as HTMLElement;

        if (target.closest('.resize-handle')) {
            isResizing.current = true;
            resizeStart.current = {
                x: e.clientX,
                y: e.clientY,
                w: size.width,
                h: size.height,
            };
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            e.preventDefault();
        } else if (target.closest('.drag-handle')) {
            isDragging.current = true;
            dragOffset.current = {
                x: e.clientX - position.x,
                y: e.clientY - position.y,
            };
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }
    };

    const handleMouseMove = (e: MouseEvent) => {
        if (isDragging.current) {
            setPosition({
                x: e.clientX - dragOffset.current.x,
                y: e.clientY - dragOffset.current.y,
            });
        } else if (isResizing.current) {
            const dw = e.clientX - resizeStart.current.x;
            const dh = e.clientY - resizeStart.current.y;
            setSize({
                width: Math.max(400, resizeStart.current.w + dw),
                height: Math.max(300, resizeStart.current.h + dh),
            });
        }
    };

    const handleMouseUp = () => {
        isDragging.current = false;
        isResizing.current = false;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return `${date.getMonth() + 1}/${date.getDate()}`;
    };

    const formatPrice = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(value);
    };

    const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold mb-1">{data.date}</p>
                    <p className="text-xs text-primary">
                        Close: {formatPrice(data.close)} VND
                    </p>
                </div>
            );
        }
        return null;
    };

    const syncBanner = priceData?.sync_error
        ? {
            className: 'alert alert-warning mb-3 py-2',
            text: `Background sync issue: ${priceData.sync_error}`,
        }
        : priceData?.sync_timed_out
            ? {
                className: 'alert alert-info mb-3 py-2',
                text: 'Showing cached data while background sync continues.',
            }
            : null;

    return (
        <div
            ref={popupRef}
            className="fixed card bg-base-100 shadow-2xl border border-base-300 overflow-hidden flex flex-col price-chart-popup"
            style={{
                left: position.x,
                top: position.y,
                width: size.width,
                height: size.height,
                zIndex: zIndex,
            }}
            onMouseDown={() => onFocus()}
        >
            {/* Header / Drag Handle */}
            <div
                className="card-title bg-primary text-primary-content p-3 cursor-move drag-handle flex justify-between items-center shrink-0"
                onMouseDown={handleMouseDown}
            >
                <div className="flex flex-col">
                    <span className="text-sm font-bold uppercase">{ticker} - {selectedRangeDays}-Day (Calendar) Price Chart</span>
                    <span className="text-xs opacity-80 font-normal">{companyName}</span>
                </div>
                <button
                    className="btn btn-circle btn-xs btn-ghost text-primary-content"
                    onClick={(e) => {
                        e.stopPropagation();
                        onClose();
                    }}
                >
                    ✕
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden relative bg-base-100 p-4">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <span className="loading loading-spinner loading-lg text-primary"></span>
                    </div>
                ) : error ? (
                    <div className="alert alert-error">
                        <span>{error}</span>
                    </div>
                ) : priceData && priceData.data.length > 0 ? (
                    <div className="h-full flex flex-col">
                        {syncBanner ? (
                            <div className={syncBanner.className}>
                                <span className="text-xs">{syncBanner.text}</span>
                            </div>
                        ) : null}
                        <div className="flex-1 min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={priceData.data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                                    <XAxis
                                        dataKey="date"
                                        tickFormatter={formatDate}
                                        tick={{ fontSize: 12 }}
                                        stroke="currentColor"
                                        opacity={0.5}
                                    />
                                    <YAxis
                                        tickFormatter={formatPrice}
                                        tick={{ fontSize: 12 }}
                                        stroke="currentColor"
                                        opacity={0.5}
                                    />
                                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                                    <Line
                                        type="monotone"
                                        dataKey="close"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        dot={false}
                                        connectNulls={true}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full text-base-content/50">
                        No price data available
                    </div>
                )}
            </div>

            <div className="border-t border-base-300 bg-base-100 px-4 py-2 shrink-0 flex items-center justify-between gap-2">
                <div className="text-xs text-base-content/70">Date range</div>
                <div className="join">
                    {PRICE_RANGE_OPTIONS.map((option) => (
                        <button
                            key={option.days}
                            type="button"
                            className={`join-item btn btn-xs ${selectedRangeDays === option.days ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setSelectedRangeDays(option.days)}
                            disabled={loading}
                            title={`${option.days} calendar days`}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="p-2 border-t border-base-300 bg-base-200 text-[10px] text-base-content/50 text-right shrink-0 relative">
                Daily close price (VND)
                {/* Resize Handle */}
                <div
                    className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize resize-handle flex items-end justify-end p-0.5"
                    onMouseDown={handleMouseDown}
                >
                    <svg width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg" className="opacity-40">
                        <path d="M1 7L7 1M4 7L7 4M7 7V7.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                </div>
            </div>
        </div>
    );
};
