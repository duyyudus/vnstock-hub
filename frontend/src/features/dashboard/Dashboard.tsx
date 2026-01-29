import React, { useState, useEffect } from 'react';
import TabNavigation from '../../components/TabNavigation';
import IndicesTab from './indices/IndicesTab';
import IndexBanners from './banner/IndexBanners';
import { stockApi } from '../../api/stockApi';
import type { IndexConfig } from './indices/indexConfig';
import { FundsTab } from './funds/FundsTab';
import { AuthWidget } from '../auth/AuthWidget';

// Tab definitions
const DASHBOARD_TABS = [
    { id: 'indices', label: 'Indices' },
    { id: 'funds', label: 'Funds' },
];

import { CompanyFinancialPopup } from './components/CompanyFinancialPopup';
import { VolumeChartPopup } from './components/VolumeChartPopup';

interface OpenPopup {
    ticker: string;
    companyName: string;
    position: { x: number; y: number };
    zIndex: number;
}

interface OpenVolumePopup {
    ticker: string;
    companyName: string;
    position: { x: number; y: number };
    zIndex: number;
}

/**
 * Main dashboard component with tab navigation.
 */
export const Dashboard: React.FC = () => {
    const [activeTab, setActiveTab] = useState('indices');
    const [indices, setIndices] = useState<IndexConfig[]>([]);
    const [loadingIndices, setLoadingIndices] = useState(true);
    const [openPopups, setOpenPopups] = useState<OpenPopup[]>([]);
    const [openVolumePopups, setOpenVolumePopups] = useState<OpenVolumePopup[]>([]);
    const [maxZIndex, setMaxZIndex] = useState(100);



    useEffect(() => {
        // Expose handleTickerClick to global window for StocksTable to use
        // This is a workaround to avoid passing props deep or using Context for now
        (window as any).onTickerClick = (ticker: string, companyName: string) => {
            handleTickerClick(ticker, companyName);
        };
        (window as any).onVolumeClick = (ticker: string, companyName: string) => {
            handleVolumeClick(ticker, companyName);
        };
        return () => {
            delete (window as any).onTickerClick;
            delete (window as any).onVolumeClick;
        };
    }, [openPopups, openVolumePopups, maxZIndex]);

    const handleTickerClick = (ticker: string, companyName: string) => {
        // Check if already open
        if (openPopups.find(p => p.ticker === ticker)) {
            // Focus it instead
            focusPopup(ticker);
            return;
        }

        const newZIndex = maxZIndex + 1;
        const offset = openPopups.length * 30;
        const newPopup: OpenPopup = {
            ticker,
            companyName,
            position: { x: 100 + offset, y: 100 + offset },
            zIndex: newZIndex,
        };

        setOpenPopups([...openPopups, newPopup]);
        setMaxZIndex(newZIndex);
    };

    const closePopup = (ticker: string) => {
        setOpenPopups(openPopups.filter(p => p.ticker !== ticker));
    };

    const focusPopup = (ticker: string) => {
        const newZIndex = maxZIndex + 1;
        setOpenPopups(openPopups.map(p =>
            p.ticker === ticker ? { ...p, zIndex: newZIndex } : p
        ));
        setMaxZIndex(newZIndex);
    };

    const handleVolumeClick = (ticker: string, companyName: string) => {
        // Check if already open
        if (openVolumePopups.find(p => p.ticker === ticker)) {
            // Focus it instead
            focusVolumePopup(ticker);
            return;
        }

        const newZIndex = maxZIndex + 1;
        const offset = (openPopups.length + openVolumePopups.length) * 30;
        const newPopup: OpenVolumePopup = {
            ticker,
            companyName,
            position: { x: 150 + offset, y: 150 + offset },
            zIndex: newZIndex,
        };

        setOpenVolumePopups([...openVolumePopups, newPopup]);
        setMaxZIndex(newZIndex);
    };

    const closeVolumePopup = (ticker: string) => {
        setOpenVolumePopups(openVolumePopups.filter(p => p.ticker !== ticker));
    };

    const focusVolumePopup = (ticker: string) => {
        const newZIndex = maxZIndex + 1;
        setOpenVolumePopups(openVolumePopups.map(p =>
            p.ticker === ticker ? { ...p, zIndex: newZIndex } : p
        ));
        setMaxZIndex(newZIndex);
    };

    useEffect(() => {
        const fetchIndices = async () => {
            try {
                const response = await stockApi.getIndices();
                const mappedIndices: IndexConfig[] = response.indices.map((idx) => ({
                    id: idx.symbol,
                    label: idx.symbol,
                    title: idx.name,
                    description: idx.description || `Stocks in ${idx.name}`,
                    apiEndpoint: idx.symbol, // Use symbol as ID for generic endpoint
                }));
                setIndices(mappedIndices);
            } catch (error) {
                console.error('Failed to fetch indices:', error);
                // Fallback to empty list or static default if needed
            } finally {
                setLoadingIndices(false);
            }
        };

        fetchIndices();
    }, []);

    // Render content based on active tab
    const renderContent = () => {
        switch (activeTab) {
            case 'indices':
                if (loadingIndices) {
                    return (
                        <div className="flex flex-col items-center justify-center h-64">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                            <p className="mt-4 text-base-content/70">Loading available indices...</p>
                        </div>
                    );
                }
                return <IndicesTab indices={indices} />;
            case 'funds':
                return <FundsTab />;
            default:
                return (
                    <div className="flex items-center justify-center h-64">
                        <p className="text-base-content/60">Select a tab to view content</p>
                    </div>
                );
        }
    };

    return (
        <div className="min-h-screen bg-base-300">
            {/* Header */}
            <header className="navbar bg-base-100 shadow-lg px-4 md:px-6">
                <div className="max-w-[96.8rem] mx-auto w-full flex items-center">
                    <div className="flex-1">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                            🚀 VNStock Hub
                        </h1>
                    </div>
                    <div className="flex-none flex items-center gap-2">
                        <AuthWidget />
                        <div className="dropdown dropdown-end">
                            <label tabIndex={0} className="btn btn-ghost btn-circle">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="h-5 w-5"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="2"
                                        d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                                    />
                                </svg>
                            </label>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main content with sidebar centered */}
            <div className="max-w-[96.8rem] mx-auto w-full p-6">
                {/* Index Banners Row */}
                <IndexBanners />

                <div className="flex gap-6">
                    {/* Left sidebar - Tab navigation */}
                    <aside className="shrink-0">
                        <TabNavigation
                            tabs={DASHBOARD_TABS}
                            activeTab={activeTab}
                            onTabChange={setActiveTab}
                        />
                    </aside>

                    {/* Right content area */}
                    <main className="flex-1 bg-base-100 rounded-xl p-6 shadow-lg overflow-hidden">
                        {renderContent()}
                    </main>
                </div>
            </div>

            {/* Financial Details Popups */}
            {openPopups.map((popup) => (
                <CompanyFinancialPopup
                    key={popup.ticker}
                    ticker={popup.ticker}
                    companyName={popup.companyName}
                    initialPosition={popup.position}
                    zIndex={popup.zIndex}
                    onClose={() => closePopup(popup.ticker)}
                    onFocus={() => focusPopup(popup.ticker)}
                />
            ))}

            {/* Volume Chart Popups */}
            {openVolumePopups.map((popup) => (
                <VolumeChartPopup
                    key={`volume-${popup.ticker}`}
                    ticker={popup.ticker}
                    companyName={popup.companyName}
                    initialPosition={popup.position}
                    zIndex={popup.zIndex}
                    onClose={() => closeVolumePopup(popup.ticker)}
                    onFocus={() => focusVolumePopup(popup.ticker)}
                />
            ))}
        </div>
    );
};

export default Dashboard;
