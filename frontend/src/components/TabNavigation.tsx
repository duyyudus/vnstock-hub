import React from 'react';

interface TabItem {
    id: string;
    label: string;
    icon?: React.ReactNode;
}

interface TabNavigationProps {
    tabs: TabItem[];
    activeTab: string;
    onTabChange: (tabId: string) => void;
}

/**
 * Vertical tab navigation component using daisyUI styling.
 */
export const TabNavigation: React.FC<TabNavigationProps> = ({
    tabs,
    activeTab,
    onTabChange,
}) => {
    return (
        <nav className="flex flex-col gap-2 min-w-[152px] bg-base-200 rounded-xl p-2.5">
            <ul className="menu menu-vertical gap-1">
                {tabs.map((tab) => (
                    <li key={tab.id}>
                        <button
                            className={`btn btn-ghost justify-start gap-2.5 px-3 ${activeTab === tab.id
                                ? 'btn-active bg-primary text-white'
                                : 'hover:bg-base-300'
                                }`}
                            onClick={() => onTabChange(tab.id)}
                        >
                            {tab.icon && <span className="text-xl">{tab.icon}</span>}
                            <span className="font-medium">{tab.label}</span>
                        </button>
                    </li>
                ))}
            </ul>
        </nav>
    );
};

export default TabNavigation;
