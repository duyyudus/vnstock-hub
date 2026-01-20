import React from 'react';
import type { IndexConfig } from './indexConfig';

interface IndexSelectorProps {
    /** List of available indices */
    indices: IndexConfig[];
    /** Currently selected index */
    selectedIndex: IndexConfig;
    /** Callback when user selects a different index */
    onIndexChange: (index: IndexConfig) => void;
}

/**
 * Dropdown selector for choosing which stock index to display.
 * Positioned in the top-right corner of the index table header.
 */
export const IndexSelector: React.FC<IndexSelectorProps> = ({
    indices,
    selectedIndex,
    onIndexChange,
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newIndex = indices.find((idx) => idx.id === event.target.value);
        if (newIndex) {
            onIndexChange(newIndex);
        }
    };

    return (
        <select
            className="select select-bordered select-sm w-40 bg-base-100 font-medium"
            value={selectedIndex.id}
            onChange={handleChange}
            aria-label="Select stock index"
        >
            {indices.map((index) => (
                <option key={index.id} value={index.id}>
                    {index.label}
                </option>
            ))}
        </select>
    );
};

export default IndexSelector;
