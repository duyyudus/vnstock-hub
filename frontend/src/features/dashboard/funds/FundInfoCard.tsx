import React from 'react';

interface FundInfoCardProps {
    fundInfo: any | null;
    loading?: boolean;
}

/**
 * Card component displaying basic fund information.
 */
export const FundInfoCard: React.FC<FundInfoCardProps> = ({ fundInfo, loading = false }) => {
    if (loading) {
        return (
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4">
                    <div className="flex items-center justify-center h-20">
                        <span className="loading loading-spinner loading-md"></span>
                    </div>
                </div>
            </div>
        );
    }

    if (!fundInfo) {
        return (
            <div className="card bg-base-100 shadow-md border border-base-300">
                <div className="card-body p-4">
                    <div className="text-center text-base-content/50">
                        Select a fund to view details
                    </div>
                </div>
            </div>
        );
    }

    // Extract fund information from the data object
    const symbol = fundInfo.symbol || fundInfo.fund_code || 'N/A';
    const name = fundInfo.fund_name || fundInfo.name || 'N/A';
    const fundType = fundInfo.fund_type || fundInfo.type || 'N/A';
    const fundOwner = fundInfo.fund_owner || fundInfo.owner || fundInfo.management_company || 'N/A';
    const nav = fundInfo.nav || fundInfo.net_asset_value || null;
    const fee = fundInfo.management_fee || fundInfo.fee || null;
    const inceptionDate = fundInfo.inception_date || fundInfo.start_date || null;

    return (
        <div className="card bg-base-100 shadow-md border border-base-300">
            <div className="card-body p-4">
                <div className="grid grid-cols-2 md:grid-cols-4 lg:flex lg:flex-row lg:flex-wrap lg:items-start lg:justify-between lg:gap-x-12 gap-4">
                    <div className="flex-shrink-0">
                        <div className="text-xs text-base-content/60 mb-1">Symbol</div>
                        <div className="font-semibold text-sm">{symbol}</div>
                    </div>
                    <div className="md:col-span-2 flex-shrink-0">
                        <div className="text-xs text-base-content/60 mb-1">Name</div>
                        <div className="font-semibold text-sm whitespace-nowrap" title={name}>
                            {name}
                        </div>
                    </div>
                    <div className="flex-shrink-0">
                        <div className="text-xs text-base-content/60 mb-1">Type</div>
                        <div className="font-semibold text-sm whitespace-nowrap" title={fundType}>
                            {fundType}
                        </div>
                    </div>
                    <div className="md:col-span-2 flex-shrink-0">
                        <div className="text-xs text-base-content/60 mb-1">Owner</div>
                        <div className="font-semibold text-sm whitespace-nowrap" title={fundOwner}>
                            {fundOwner}
                        </div>
                    </div>
                    {nav !== null && (
                        <div className="flex-shrink-0">
                            <div className="text-xs text-base-content/60 mb-1">NAV</div>
                            <div className="font-semibold text-sm">
                                {typeof nav === 'number' ? nav.toLocaleString() : nav}
                            </div>
                        </div>
                    )}
                    {fee !== null && (
                        <div className="flex-shrink-0">
                            <div className="text-xs text-base-content/60 mb-1">Fee</div>
                            <div className="font-semibold text-sm">
                                {typeof fee === 'number' ? `${fee}%` : fee}
                            </div>
                        </div>
                    )}
                    {inceptionDate && (
                        <div className="flex-shrink-0">
                            <div className="text-xs text-base-content/60 mb-1">Since</div>
                            <div className="font-semibold text-sm">{inceptionDate}</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FundInfoCard;
