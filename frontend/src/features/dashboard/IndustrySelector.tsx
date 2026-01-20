import React from 'react';
import type { IndustryInfo } from '../../api/stockApi';

interface IndustrySelectorProps {
    /** List of available industries */
    industries: IndustryInfo[];
    /** Currently selected industry name */
    selectedIndustryName: string | null;
    /** Callback when user selects a different industry */
    onIndustryChange: (industryName: string | null) => void;
}

/**
 * Dropdown selector for choosing which stock industry to display.
 */
export const IndustrySelector: React.FC<IndustrySelectorProps> = ({
    industries,
    selectedIndustryName,
    onIndustryChange,
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const val = event.target.value;
        onIndustryChange(val === "" ? null : val);
    };

    return (
        <select
            className="select select-bordered select-sm w-48 bg-base-100 font-medium"
            value={selectedIndustryName || ""}
            onChange={handleChange}
            aria-label="Select stock industry"
        >
            <option value="">-- All Industries --</option>
            {industries.map((industry) => (
                <option key={industry.code} value={industry.name}>
                    {industry.name}
                </option>
            ))}
        </select>
    );
};

export default IndustrySelector;
