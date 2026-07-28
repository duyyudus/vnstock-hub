interface ForeignHoldingInput {
    current_room: number | null | undefined;
    total_room: number | null | undefined;
    price: number | null | undefined;
}

export interface ForeignHoldingInfo {
    shares: number;
    valueBilVnd: number;
}

export const getForeignHolding = ({
    current_room: currentRoom,
    total_room: totalRoom,
    price,
}: ForeignHoldingInput): ForeignHoldingInfo | null => {
    if (
        currentRoom == null
        || totalRoom == null
        || price == null
        || !Number.isFinite(currentRoom)
        || !Number.isFinite(totalRoom)
        || !Number.isFinite(price)
        || currentRoom < 0
        || totalRoom <= 0
        || currentRoom > totalRoom
        || price < 0
    ) {
        return null;
    }

    const shares = totalRoom - currentRoom;
    return {
        shares,
        valueBilVnd: (shares * price) / 1e9,
    };
};
