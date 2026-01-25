import React from 'react';

interface SyncIndicatorProps {
    isVisible: boolean;
    tooltip?: string;
}

/**
 * Spinning indicator shown in header when background sync is in progress.
 */
export const SyncIndicator: React.FC<SyncIndicatorProps> = ({
    isVisible,
    tooltip = "Syncing latest fund data..."
}) => {
    if (!isVisible) return null;

    return (
        <div className="tooltip tooltip-left" data-tip={tooltip}>
            <div className="flex items-center gap-2 px-2 py-1 rounded-lg bg-primary/10">
                <span className="loading loading-spinner loading-sm text-primary"></span>
                <span className="text-xs text-primary font-medium hidden sm:inline">Syncing</span>
            </div>
        </div>
    );
};

export default SyncIndicator;
