import React from 'react';
import type { BookmarkGroup } from '../../../api/stockApi';

interface BookmarkSelectorProps {
    groups: BookmarkGroup[];
    selectedGroupId: number | null;
    onGroupChange: (groupId: number | null) => void;
    disabled?: boolean;
}

export const BookmarkSelector: React.FC<BookmarkSelectorProps> = ({
    groups,
    selectedGroupId,
    onGroupChange,
    disabled = false,
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const value = event.target.value;
        onGroupChange(value === '' ? null : Number(value));
    };

    return (
        <select
            className="select select-bordered select-sm w-48 bg-base-100 font-medium"
            value={selectedGroupId ?? ''}
            onChange={handleChange}
            aria-label="Select bookmark group"
            disabled={disabled}
        >
            <option value="">-- Bookmarks --</option>
            {groups.length === 0 ? (
                <option value="" disabled>
                    No groups yet
                </option>
            ) : (
                groups.map((group) => (
                    <option key={group.id} value={group.id}>
                        {group.name}
                    </option>
                ))
            )}
        </select>
    );
};

export default BookmarkSelector;
