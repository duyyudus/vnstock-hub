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

type TabType = 'overview' | 'income' | 'balance' | 'cashflow' | 'ratios' | 'shareholders' | 'officers' | 'subsidiaries';

export const CompanyFinancialPopup: React.FC<CompanyFinancialPopupProps> = ({
    ticker,
    companyName,
    initialPosition,
    onClose,
    zIndex,
    onFocus,
}) => {
    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<any[]>([]);
    const [position, setPosition] = useState<Position>(initialPosition);
    const [size, setSize] = useState<Size>({ width: 900, height: 550 });
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
                    case 'overview':
                        response = await stockApi.getCompanyOverview(ticker);
                        break;
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
                    case 'shareholders':
                        response = await stockApi.getShareholders(ticker);
                        break;
                    case 'officers':
                        response = await stockApi.getOfficers(ticker);
                        break;
                    case 'subsidiaries':
                        response = await stockApi.getSubsidiaries(ticker);
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

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                // Find all open popups
                const popups = Array.from(document.querySelectorAll('.company-financial-popup')) as HTMLElement[];
                if (popups.length === 0) return;

                // Find the one with highest z-index
                let highestZ = -1;
                popups.forEach(p => {
                    const z = parseInt(p.style.zIndex || '0', 10);
                    if (z > highestZ) highestZ = z;
                });

                // Only close if this instance has the highest z-index
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
    const formatValue = (val: any, key?: string) => {
        if (val === null || val === undefined) return '-';
        const lowerKey = (key || '').toLowerCase();

        const getFormattedValue = () => {
            if (typeof val === 'number') {
                // Handle percentages (0.15 -> 15%)
                if (lowerKey.includes('percent')) {
                    return new Intl.NumberFormat('en-US', {
                        style: 'percent',
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                    }).format(val);
                }

                // Handle quantities (large integers)
                if (lowerKey.includes('quantity') || lowerKey.includes('volume') || lowerKey.includes('share')) {
                    const absVal = Math.abs(val);
                    if (absVal >= 1e9) {
                        return (val / 1e9).toFixed(3) + ' B';
                    }
                    if (absVal >= 1e6) {
                        return (val / 1e6).toFixed(3) + ' M';
                    }
                    return new Intl.NumberFormat('en-US').format(val);
                }

                // Standard financial values (Income, Balance, etc.)
                // Check if it's potentially VND (very large) or ratio (small)
                if (Math.abs(val) > 1e6) { // Most financial values are in Bn VND or large VND
                    return new Intl.NumberFormat('en-US').format(Math.round(val / 1e6) / 1000);
                }

                return new Intl.NumberFormat('en-US', {
                    maximumFractionDigits: 3,
                }).format(val);
            }
            return String(val);
        };

        const result = getFormattedValue();
        if (lowerKey === 'charter_capital') {
            return `${result} Bil VND`;
        }
        return result;
    };

    // Render Overview tab with a nice layout
    const renderOverview = () => {
        if (data.length === 0) return <div className="p-8 text-center text-base-content/50">No data available</div>;

        const item = data[0];

        // Known long text fields
        const longTextFields = ['company_profile', 'business_strategy', 'key_developments', 'history_dev', 'history', 'company_promise'];

        // Financial or numeric fields for formatting
        const numericFields = ['charter_capital', 'listing_volume', 'foreign_ownership_ratio'];

        // Filter and group keys
        const keys = Object.keys(item).filter(key =>
            !['ticker', 'Meta_ticker', 'id', 'Meta_yearReport', 'Meta_lengthReport'].includes(key)
        );

        const sections = {
            general: keys.filter(k => !longTextFields.includes(k)),
            detailed: keys.filter(k => longTextFields.includes(k)),
        };

        const row1Keys = ['symbol', 'issue_share', 'financial_ratio_issue_share', 'charter_capital'];
        const row2Keys = ['icb_name1', 'icb_name2', 'icb_name3', 'icb_name4'];
        const specialFields = [...row1Keys, ...row2Keys];

        const remainingGeneral = sections.general.filter((k: string) => !specialFields.includes(k));

        const renderField = (key: string) => (
            <div key={key} className="flex flex-col group transition-all duration-200">
                <span className="text-[10px] uppercase font-bold text-base-content/50 mb-1 group-hover:text-primary transition-colors">
                    {key.replace(/_/g, ' ')}
                </span>
                <span className="text-sm font-medium text-base-content/90 break-words">
                    {(numericFields.includes(key) || key.includes('share'))
                        ? formatValue(item[key], key)
                        : String(item[key] || '-')}
                </span>
            </div>
        );

        const renderDetailedSection = (key: string) => {
            const content = item[key];
            const isHistory = key === 'history' || key === 'history_dev';

            return (
                <section key={key} className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                    <h3 className="text-sm font-bold uppercase tracking-wider mb-4 border-b border-base-300 pb-2 text-primary flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                        </svg>
                        {key.replace(/_/g, ' ')}
                    </h3>
                    <div className="text-sm leading-relaxed text-base-content/80 whitespace-pre-wrap bg-base-200/40 p-5 rounded-2xl border border-base-300/50 shadow-inner">
                        {isHistory && typeof content === 'string' ? (
                            <div className="space-y-3">
                                {content.split('-').filter(s => s.trim()).map((part, idx) => (
                                    <div key={idx} className="flex gap-3 items-start group/item">
                                        <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-primary/40 group-hover/item:bg-primary transition-colors flex-shrink-0" />
                                        <div className="flex-1">{part.trim()}</div>
                                    </div>
                                ))}
                            </div>
                        ) : content}
                    </div>
                </section>
            );
        };

        return (
            <div className="p-6 overflow-auto h-full space-y-8 bg-base-100 custom-scrollbar">
                {/* Company Profile - Priority 1 */}
                {item['company_profile'] && renderDetailedSection('company_profile')}

                {/* General Information - Priority 2 */}
                {sections.general.length > 0 && (
                    <section>
                        <h3 className="text-sm font-bold uppercase tracking-wider mb-4 border-b border-base-300 pb-2 text-primary flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            General Information
                        </h3>

                        <div className="space-y-6">
                            {/* Row 1: Symbol & Shares */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 bg-base-200/30 p-4 rounded-xl border border-base-300/30">
                                {row1Keys.map(k => item[k] !== undefined && renderField(k))}
                            </div>

                            {/* Row 2: ICB Hierarchy */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 bg-base-200/30 p-4 rounded-xl border border-base-300/30">
                                {row2Keys.map(k => item[k] !== undefined && renderField(k))}
                            </div>

                            {/* Remaining General Info */}
                            {remainingGeneral.length > 0 && (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-2">
                                    {remainingGeneral.map((k: string) => renderField(k))}
                                </div>
                            )}
                        </div>
                    </section>
                )}

                {/* Other Detailed Sections - Priority 3 */}
                {sections.detailed
                    .filter(k => k !== 'company_profile')
                    .map((key: string) => renderDetailedSection(key))}
            </div>
        );
    };

    // Render as a simple list for non-financial statement data
    const renderListTable = () => {
        if (data.length === 0) return <div className="p-8 text-center text-base-content/50">No data available</div>;

        const allKeys = Array.from(new Set(data.flatMap(item => Object.keys(item))))
            .filter(key => !['ticker', 'Meta_ticker', 'id'].includes(key));

        return (
            <div className="overflow-auto h-full">
                <table className="table table-xs table-pin-rows">
                    <thead>
                        <tr>
                            {allKeys.map(key => (
                                <th key={key} className="bg-base-200 capitalize">
                                    {key.replace(/_/g, ' ')}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((item, i) => (
                            <tr key={i} className="hover">
                                {allKeys.map(key => (
                                    <td key={key} className="whitespace-nowrap font-mono text-xs">
                                        {formatValue(item[key], key)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
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
                                        {formatValue(item[key], key)}
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
            className="fixed card bg-base-100 shadow-2xl border border-base-300 overflow-hidden flex flex-col company-financial-popup"
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
                    className={`tab tab-sm flex-1 ${activeTab === 'overview' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
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
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'shareholders' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('shareholders')}
                >
                    Shareholders
                </button>
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'officers' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('officers')}
                >
                    Officers
                </button>
                <button
                    className={`tab tab-sm flex-1 ${activeTab === 'subsidiaries' ? 'tab-active' : ''}`}
                    onClick={() => setActiveTab('subsidiaries')}
                >
                    Subsidiaries
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
                    activeTab === 'overview' ? renderOverview() :
                        ['shareholders', 'officers', 'subsidiaries'].includes(activeTab)
                            ? renderListTable()
                            : renderTable()
                )}
            </div>

            <div className="p-2 border-t border-base-300 bg-base-200 text-[10px] text-base-content/50 text-right shrink-0 relative">
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
