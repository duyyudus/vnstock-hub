import React from 'react';
import type { FundInfo } from './FundSelector';

type FundDataValue = string | number | boolean | null;
type FundDataRecord = Record<string, FundDataValue>;
type FundInfoSource = FundInfo | FundDataRecord;

interface FundInfoCardProps {
    fundInfo: FundInfoSource | null;
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

    const getStringValue = (...values: Array<FundDataValue | undefined>) => {
        for (const value of values) {
            if (typeof value === 'string' && value.trim()) {
                return value;
            }
        }
        return null;
    };

    const getNumberValue = (...values: Array<FundDataValue | undefined>) => {
        for (const value of values) {
            if (typeof value === 'number') {
                return value;
            }
        }
        return null;
    };

    // Extract fund information from the data object
    const symbol = getStringValue(fundInfo.symbol, 'fund_code' in fundInfo ? fundInfo.fund_code : null) || 'N/A';
    const name = getStringValue('fund_name' in fundInfo ? fundInfo.fund_name : null, fundInfo.name) || 'N/A';
    const fundType = getStringValue(
        'fund_type' in fundInfo ? fundInfo.fund_type : null,
        'type' in fundInfo ? fundInfo.type : null,
    ) || 'N/A';
    const fundOwner = getStringValue(
        'fund_owner' in fundInfo ? fundInfo.fund_owner : null,
        'owner' in fundInfo ? fundInfo.owner : null,
        'management_company' in fundInfo ? fundInfo.management_company : null,
    ) || 'N/A';
    const nav = getNumberValue(
        'nav' in fundInfo ? fundInfo.nav : null,
        'net_asset_value' in fundInfo ? fundInfo.net_asset_value : null,
    );
    const fee = getNumberValue(
        'management_fee' in fundInfo ? fundInfo.management_fee : null,
        'fee' in fundInfo ? fundInfo.fee : null,
    );
    const inceptionDate = getStringValue(
        'inception_date' in fundInfo ? fundInfo.inception_date : null,
        'start_date' in fundInfo ? fundInfo.start_date : null,
    );

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
