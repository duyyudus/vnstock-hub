import React, { useState, useEffect, useMemo, useRef } from 'react';
import { stockApi } from '../../../api/stockApi';
import type { VolumeDataPoint, VolumeHistoryResponse } from '../../../api/stockApi';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

const VOLUME_RANGE_OPTIONS = [
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

interface VolumeChartPopupProps {
    ticker: string;
    companyName: string;
    initialPosition: Position;
    onClose: () => void;
    zIndex: number;
    onFocus: () => void;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: Array<{ payload: ChartPoint }>;
}

interface ChartPoint extends VolumeDataPoint {
    display_value: number | null;
    display_volume: number;
    using_total_value: boolean;
    using_total_volume: boolean;
}

const formatDate = (dateStr: string) => {
    const date = new Date(`${dateStr}T00:00:00`);
    if (Number.isNaN(date.getTime())) {
        return dateStr;
    }
    return `${date.getMonth() + 1}/${date.getDate()}`;
};

const formatVolume = (value: number) => {
    if (value >= 1e6) {
        return `${(value / 1e6).toFixed(1)}M`;
    }
    if (value >= 1e3) {
        return `${(value / 1e3).toFixed(1)}K`;
    }
    return value.toString();
};

const formatCompactBilVnd = (value: number) => {
    const absoluteValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    if (absoluteValue >= 1000) {
        return `${sign}${(absoluteValue / 1000).toFixed(1)}K`;
    }
    if (absoluteValue >= 100) {
        return `${sign}${absoluteValue.toFixed(0)}`;
    }
    if (absoluteValue >= 10) {
        return `${sign}${absoluteValue.toFixed(1)}`;
    }
    return `${sign}${absoluteValue.toFixed(2)}`;
};

const formatValue = (value: number | null, signed: boolean = false) => {
    if (value === null) return 'N/A';
    const prefix = signed && value > 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)} B VND`;
};

const formatImpactRatio = (flowValue: number | null, totalValue: number | null) => {
    if (flowValue === null || totalValue === null || totalValue === 0) {
        return null;
    }

    const ratio = (flowValue / totalValue) * 100;
    const prefix = ratio > 0 ? '+' : '';
    return `${prefix}${ratio.toFixed(1)}% of total`;
};

export const VolumeChartPopup: React.FC<VolumeChartPopupProps> = ({
    ticker,
    companyName,
    initialPosition,
    onClose,
    zIndex,
    onFocus,
}) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [volumeData, setVolumeData] = useState<VolumeHistoryResponse | null>(null);
    const [selectedRangeDays, setSelectedRangeDays] = useState(90);
    const [position, setPosition] = useState<Position>(initialPosition);
    const [size, setSize] = useState<Size>({ width: 1050, height: 675 });
    const isDragging = useRef(false);
    const isResizing = useRef(false);
    const dragOffset = useRef<Position>({ x: 0, y: 0 });
    const resizeStart = useRef<{ x: number, y: number, w: number, h: number }>({ x: 0, y: 0, w: 0, h: 0 });
    const popupRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchVolumeData = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await stockApi.getVolumeHistory(ticker, selectedRangeDays, { autoSync: false });
                setVolumeData(response);
            } catch (err) {
                console.error(`Error fetching volume history for ${ticker}:`, err);
                setError('Failed to load volume data.');
            } finally {
                setLoading(false);
            }
        };

        fetchVolumeData();
    }, [ticker, selectedRangeDays]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                const popups = Array.from(document.querySelectorAll('.volume-chart-popup')) as HTMLElement[];
                if (popups.length === 0) return;

                let highestZ = -1;
                popups.forEach(p => {
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

    const chartData = useMemo<ChartPoint[]>(() => {
        return (volumeData?.data ?? []).map((point) => ({
            ...point,
            display_value: point.total_value ?? point.value,
            display_volume: point.total_volume ?? point.volume,
            using_total_value: point.total_value !== null,
            using_total_volume: point.total_volume !== null,
        }));
    }, [volumeData]);

    const sharedValueDomain = useMemo<[number, number]>(() => {
        const values = chartData.flatMap((point) => [
            point.display_value,
            point.foreign_net_value,
            point.prop_net_value,
        ]).filter((value): value is number => value !== null);

        if (values.length === 0) {
            return [-1, 1];
        }

        const minValue = Math.min(...values, 0);
        const maxValue = Math.max(...values, 0);

        if (minValue === 0 && maxValue === 0) {
            return [-1, 1];
        }

        const paddedMin = minValue < 0
            ? minValue * 1.05
            : -Math.max(maxValue * 0.08, 1);
        const paddedMax = maxValue > 0
            ? maxValue * 1.05
            : Math.max(Math.abs(minValue) * 0.08, 1);

        return [paddedMin, paddedMax];
    }, [chartData]);

    const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            const foreignImpact = formatImpactRatio(data.foreign_net_value, data.display_value);
            const propImpact = formatImpactRatio(data.prop_net_value, data.display_value);

            return (
                <div className="bg-base-100 border border-base-300 p-3 rounded-lg shadow-lg">
                    <p className="text-sm font-semibold mb-1">{data.date}</p>
                    <p className="text-xs text-sky-500">
                        {data.using_total_value ? 'Total trade value' : 'Trade value (fallback)'}: {formatValue(data.display_value)}
                    </p>
                    <p className="text-xs text-emerald-500">
                        Foreign net: {formatValue(data.foreign_net_value, true)}
                    </p>
                    {foreignImpact ? (
                        <p className="text-[11px] text-base-content/60">
                            Foreign impact: {foreignImpact}
                        </p>
                    ) : null}
                    <p className="text-xs text-amber-500 mt-1">
                        Proprietary net: {formatValue(data.prop_net_value, true)}
                    </p>
                    {propImpact ? (
                        <p className="text-[11px] text-base-content/60">
                            Proprietary impact: {propImpact}
                        </p>
                    ) : null}
                    <p className="text-xs text-base-content/70 mt-2">
                        {data.using_total_volume ? 'Total volume' : 'Volume'}: {formatVolume(data.display_volume)}
                    </p>
                </div>
            );
        }
        return null;
    };

    const { hasForeignFlowData, hasPropFlowData, flowAvailabilityNote, totalValueFallbackNote } = useMemo(() => {
        const data = chartData;
        const hasForeign = data.some((point) => point.foreign_net_value !== null);
        const hasProp = data.some((point) => point.prop_net_value !== null);
        const hasTrueTotalValue = data.some((point) => point.total_value !== null);

        let note: string | null = null;
        if (!hasForeign && !hasProp) {
            note = 'Foreign and proprietary flow unavailable for this range.';
        } else if (!hasForeign) {
            note = 'Foreign flow unavailable for this range.';
        } else if (!hasProp) {
            note = 'Proprietary flow unavailable for this range.';
        }

        return {
            hasForeignFlowData: hasForeign,
            hasPropFlowData: hasProp,
            flowAvailabilityNote: note,
            totalValueFallbackNote: hasTrueTotalValue ? null : 'Turnover enrichment unavailable for this range; chart falls back to legacy trade value approximation.',
        };
    }, [chartData]);

    const syncBanner = volumeData?.sync_error
        ? {
            className: 'alert alert-warning mb-3 py-2',
            text: `Background sync issue: ${volumeData.sync_error}`,
        }
        : volumeData?.sync_timed_out
            ? {
                className: 'alert alert-info mb-3 py-2',
                text: 'Showing cached data while background sync continues.',
            }
            : null;

    return (
        <div
            ref={popupRef}
            className="fixed card bg-base-100 shadow-2xl border border-base-300 overflow-hidden flex flex-col volume-chart-popup"
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
                    <span className="text-sm font-bold uppercase">{ticker} - {selectedRangeDays}-Day (Calendar) Trade Value &amp; Net Flow</span>
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
                ) : volumeData && volumeData.data.length > 0 ? (
                    <div className="h-full flex flex-col">
                        {syncBanner ? (
                            <div className={syncBanner.className}>
                                <span className="text-xs">{syncBanner.text}</span>
                            </div>
                        ) : null}
                        {flowAvailabilityNote ? (
                            <div className="mb-2 text-[11px] text-base-content/60">
                                {flowAvailabilityNote}
                            </div>
                        ) : null}
                        {totalValueFallbackNote ? (
                            <div className="mb-2 text-[11px] text-base-content/60">
                                {totalValueFallbackNote}
                            </div>
                        ) : null}
                        <div className="flex-1 min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={chartData}
                                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                                    barCategoryGap="12%"
                                    barGap={2}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                                    <XAxis
                                        dataKey="date"
                                        tickFormatter={formatDate}
                                        tick={{ fontSize: 12 }}
                                        stroke="currentColor"
                                        opacity={0.5}
                                        minTickGap={24}
                                    />
                                    <YAxis
                                        yAxisId="left"
                                        tickFormatter={formatCompactBilVnd}
                                        tick={{ fontSize: 12 }}
                                        stroke="currentColor"
                                        opacity={0.5}
                                        width={56}
                                        domain={sharedValueDomain}
                                    />
                                    <YAxis
                                        yAxisId="right"
                                        orientation="right"
                                        tickFormatter={formatCompactBilVnd}
                                        tick={{ fontSize: 12 }}
                                        stroke="currentColor"
                                        opacity={0.5}
                                        width={56}
                                        domain={sharedValueDomain}
                                    />
                                    <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                                    <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                                    <ReferenceLine yAxisId="left" y={0} stroke="currentColor" strokeOpacity={0.15} />
                                    <ReferenceLine yAxisId="right" y={0} stroke="currentColor" strokeOpacity={0.25} />
                                    <Bar
                                        yAxisId="left"
                                        dataKey="display_value"
                                        name="Total Trade Value"
                                        fill="#3b82f6"
                                        radius={[4, 4, 0, 0]}
                                        maxBarSize={28}
                                    />
                                    {hasForeignFlowData ? (
                                        <Bar
                                            yAxisId="right"
                                            dataKey="foreign_net_value"
                                            name="Foreign Net Value"
                                            fill="#10b981"
                                            maxBarSize={18}
                                        />
                                    ) : null}
                                    {hasPropFlowData ? (
                                        <Bar
                                            yAxisId="right"
                                            dataKey="prop_net_value"
                                            name="Proprietary Net Value"
                                            fill="#f59e0b"
                                            maxBarSize={18}
                                        />
                                    ) : null}
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full text-base-content/50">
                        No trade value data available
                    </div>
                )}
            </div>

            <div className="border-t border-base-300 bg-base-100 px-4 py-2 shrink-0 flex items-center justify-between gap-2">
                <div className="text-xs text-base-content/70">Date range</div>
                <div className="join">
                    {VOLUME_RANGE_OPTIONS.map((option) => (
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
                Total trade value, foreign net flow, and proprietary net flow (Billion VND)
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
