import React, { useState, useEffect, useRef } from 'react';
import { stockApi } from '../../api/stockApi';
import type { FinancialDataResponse } from '../../api/stockApi';

interface Position {
    x: number;
    y: number;
}

interface Size {
    width: number;
    height: number;
}

interface CompanyFinancialPopupProps {
    ticker: string;
    companyName: string;
    initialPosition: Position;
    onClose: () => void;
    zIndex: number;
    onFocus: () => void;
}

type TabType = 'income' | 'balance' | 'cashflow' | 'ratios';

export const CompanyFinancialPopup: React.FC<CompanyFinancialPopupProps> = ({
    ticker,
    companyName,
    initialPosition,
    onClose,
    zIndex,
    onFocus,
}) => {
    const [activeTab, setActiveTab] = useState<TabType>('income');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<any[]>([]);
    const [position, setPosition] = useState<Position>(initialPosition);
    const [size, setSize] = useState<Size>({ width: 600, height: 550 });
    const [attributeWidth, setAttributeWidth] = useState(200);
    const isDragging = useRef(false);
    const isResizing = useRef(false);
    const isResizingColumn = useRef(false);
    const dragOffset = useRef<Position>({ x: 0, y: 0 });
    const resizeStart = useRef<{ x: number, y: number, w: number, h: number }>({ x: 0, y: 0, w: 0, h: 0 });
    const columnResizeStart = useRef<{ x: number, w: number }>({ x: 0, w: 0 });
    const popupRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                let response: FinancialDataResponse;
                switch (activeTab) {
                    case 'income':
                        response = await stockApi.getIncomeStatement(ticker);
                        break;
                    case 'balance':
                        response = await stockApi.getBalanceSheet(ticker);
                        break;
                    case 'cashflow':
                        response = await stockApi.getCashFlow(ticker);
                        break;
                    case 'ratios':
                        response = await stockApi.getFinancialRatios(ticker);
                        break;
                    default:
                        response = { symbol: ticker, data: [], count: 0 };
                }
                setData(response.data);
            } catch (err) {
                console.error(`Error fetching ${activeTab} for ${ticker}:`, err);
                setError(`Failed to load ${activeTab} data.`);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [ticker, activeTab]);

    const handleMouseDown = (e: React.MouseEvent) => {
        onFocus();
        const target = e.target as HTMLElement;

        if (target.closest('.col-resize-handle')) {
            isResizingColumn.current = true;
            columnResizeStart.current = {
                x: e.clientX,
                w: attributeWidth,
            };
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            e.preventDefault();
            e.stopPropagation();
        } else if (target.closest('.resize-handle')) {
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
        } else if (isResizingColumn.current) {
            const dx = e.clientX - columnResizeStart.current.x;
            setAttributeWidth(Math.max(100, columnResizeStart.current.w + dx));
        }
    };

    const handleMouseUp = () => {
        isDragging.current = false;
        isResizing.current = false;
        isResizingColumn.current = false;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    };

    // Format utility for numbers
    const formatValue = (val: any) => {
        if (val === null || val === undefined) return '-';
        if (typeof val === 'number') {
            // Check if it's potentially VND (very large) or ratio (small)
            if (Math.abs(val) > 1e6) { // Most financial values are in Bn VND or large VND
                return new Intl.NumberFormat('en-US').format(Math.round(val / 1e6) / 1000);
            }
            return new Intl.NumberFormat('en-US', {
                maximumFractionDigits: 2,
            }).format(val);
        }
        return String(val);
    };

    // Transpose data for table: rows are attributes, columns are periods
    const renderTable = () => {
        if (data.length === 0) return <div className="p-8 text-center text-base-content/50">No data available</div>;

        // Extract periods for headers - handle both standard and Meta_ prefixed keys
        const periods = data.map(item => {
            const year = item.yearReport ?? item.Meta_yearReport ?? 'N/A';
            const quarter = item.lengthReport ?? item.Meta_lengthReport ?? '?';
            return `${year} Q${quarter}`;
        });

        // Extract all attributes (keys except period ones and metadata)
        const excludeKeys = [
            'ticker', 'yearReport', 'lengthReport', 'period',
            'Meta_ticker', 'Meta_yearReport', 'Meta_lengthReport', 'Meta_period'
        ];
        const allKeys = Array.from(new Set(data.flatMap(item => Object.keys(item))))
            .filter(key => !excludeKeys.includes(key));

        return (
            <div className="overflow-auto h-full">
                <table className="table table-xs table-pin-rows table-pin-cols">
                    <thead>
                        <tr>
                            <th
                                className="bg-base-200 p-0 relative group"
                                style={{ width: attributeWidth, minWidth: attributeWidth }}
                            >
                                <div className="px-2 py-1 flex items-center justify-between h-full">
                                    <span>Attribute</span>
                                    {/* Column Resize Handle */}
                                    <div
                                        className="col-resize-handle absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-primary transition-colors z-10"
                                        onMouseDown={handleMouseDown}
                                    />
                                </div>
                            </th>
                            {periods.map((p, i) => (
                                <th key={i} className="bg-base-200 text-right whitespace-nowrap">{p}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {allKeys.map(key => (
                            <tr key={key} className="hover">
                                <th
                                    className="text-xs font-semibold whitespace-nowrap overflow-hidden text-ellipsis border-r border-base-300"
                                    style={{ width: attributeWidth, maxWidth: attributeWidth }}
                                    title={key.replace(/_/g, ' ')}
                                >
                                    {key.replace(/_/g, ' ')}
                                </th>
                                {data.map((item, i) => (
                                    <td key={i} className="text-right font-mono text-xs">
                                        {formatValue(item[key])}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    return (
        <div
            ref={popupRef}
            className="fixed card bg-base-100 shadow-2xl border border-base-300 overflow-hidden flex flex-col"
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
                    <span className="text-sm font-bold uppercase">{ticker} - Financial Details</span>
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

            {/* Tabs */}
            <div className="tabs tabs-boxed rounded-none bg-base-200 p-1 shrink-0">
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'income' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('income')}
                >
                    Income
                </button>
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'balance' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('balance')}
                >
                    Balance
                </button>
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'cashflow' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('cashflow')}
                >
                    Cash Flow
                </button>
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'ratios' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('ratios')}
                >
                    Ratios
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden relative bg-base-100">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <span className="loading loading-spinner loading-lg text-primary"></span>
                    </div>
                ) : error ? (
                    <div className="alert alert-error m-4">
                        <span>{error}</span>
                    </div>
                ) : (
                    renderTable()
                )}
            </div>

            <div className="p-2 border-t border-base-300 bg-base-200 text-[10px] text-base-content/50 text-right shrink-0 relative">
                Values in Billion VND where applicable
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
