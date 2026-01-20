import React, { useState } from 'react';
import TabNavigation from '../../components/TabNavigation';
import IndexTable from './IndexTable';

// Tab definitions
const DASHBOARD_TABS = [
    { id: 'indices', label: 'Indices' },
    // Future tabs can be added here
    // { id: 'watchlist', label: 'Watchlist', icon: '⭐' },
    // { id: 'portfolio', label: 'Portfolio', icon: '💼' },
];

/**
 * Main dashboard component with tab navigation.
 */
export const Dashboard: React.FC = () => {
    const [activeTab, setActiveTab] = useState('indices');

    // Render content based on active tab
    const renderContent = () => {
        switch (activeTab) {
            case 'indices':
                return <IndexTable />;
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
                <div className="max-w-7xl mx-auto w-full flex items-center">
                    <div className="flex-1">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                            🚀 VNStock Hub
                        </h1>
                    </div>
                    <div className="flex-none gap-2">
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
            <div className="max-w-7xl mx-auto w-full p-6">
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
        </div>
    );
};

export default Dashboard;
